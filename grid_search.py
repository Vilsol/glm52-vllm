#!/usr/bin/env python3
"""
grid_search.py — automated GLM-5.2 serving-config sweep for 8x RTX PRO 6000 Blackwell.

The problem this solves: finding the best vLLM/b12x launch config for GLM-5.2 means
rebooting the engine (~7 min each) for every parameter combination, then running a
correctness check and a throughput bench by hand. This harness does that loop
unattended and picks configs intelligently instead of brute-forcing a grid.

Two modes
---------
--mode static    Evaluate a fixed list of experiments (DEFAULT_GRID). Simple and
                 predictable; good for re-checking a known set.

--mode adaptive  Budget-aware adaptive search (the overnight mode):
                   Phase 1  Greedy COORDINATE DESCENT. Tune one axis at a time and
                            lock the winner before moving on: DCP-interleave -> MTP ->
                            gpu_mem, then sweep DCP. DCP is kept as a *Pareto axis*
                            (1 = fast/low-context, 2 = balanced ~1M, 4 = max-context),
                            not collapsed to a single "winner". (This sweep tunes
                            HYPERPARAMETERS only. Backend/tech toggles are tested
                            manually outside it: fuse_allreduce_rms faults on this
                            2-socket pod, and the FlashInfer SM120 backend is A/B'd
                            by hand — the attn_backend spec key still works via
                            --grid-file for that.)
                   Phase 2  EXPLORATION. Spend the remaining budget on local
                            perturbations of the Pareto winners plus random configs.
                   Phase 3  CONFIRMATION. Re-measure the Pareto front with more
                            Estonia runs and a longer bench for stable numbers.
                 Objective is multi-objective: maximize {decode_c8, prefill_8k,
                 kv_cache_tokens} subject to the Estonia correctness gate.

Why coordinate descent and not Bayesian optimization / Hyperband: the cost here is
dominated by the per-config *boot*, which can't be amortized (one model fills all 8
GPUs) and is paid even by a "cheap" fidelity rung — so multi-fidelity buys little.
The win is minimizing how many configs we boot at all, which a greedy descent over a
small, well-understood, mostly-categorical space does well at ~40-50 evals.

Per config the harness: teardown -> launch.sh (auto-pull image if missing) ->
wait /health -> Estonia correctness gate -> trimmed throughput bench -> parse both
result JSONs -> append one CSV row -> teardown. Each config is crash-isolated (any
boot/Estonia/bench/timeout failure is recorded and the sweep continues), the run is
resumable by parameter-signature (a prior CSV row is reused instead of re-booted),
and it is deadline-aware (stops launching in time to finish confirmation).

Examples
--------
  uv run python3 grid_search.py --mode adaptive --max-hours 14.5     # the overnight run
  uv run python3 grid_search.py --self-test                          # validate logic, no Docker
  uv run python3 grid_search.py --mode static --dry-run
  uv run python3 grid_search.py --mode adaptive --only-phase coord   # coordinate descent only
"""
import argparse
import csv
import json
import os
import random
import re
import signal
import subprocess
import time
from datetime import datetime

# --- fixed environment (this host / repo layout) -----------------------------
GLM_DIR   = "/root/glm52-vllm"
LAUNCH_SH = os.path.join(GLM_DIR, "launch.sh")   # the proven launcher (host NCCL/b12x fixes live here)
BENCH_DIR = "/root/llm-inference-bench"
BENCH_PY  = "llm_decode_bench.py"
NAME      = "glm52"     # container name; must match launch.sh's default
PORT      = 8080
MODEL     = "GLM-5.2"   # served-model-name

# The two image generations we compare. v12 "dark-devotion" is the retired
# production image; v13 "eldritch" is its successor. The v13 tag here is the
# 20260629 "enlightenment" bugfix build (canonical per glm5.2_v13.md): the 20260625
# launch build had GPU-CPU sync regressions + random OOMs, 20260627 fixed a
# topk_scores_buffer crash, and 20260629 reclaims KV cache lost to head padding.
V12_IMAGE = "voipmonitor/vllm:glm52-dark-devotion-release-vllmec65667-b12xaaf1891-scale-fix-cu132-20260622"
V13_IMAGE = "voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x8ce61f9-cu132-20260629"

# v13's alternative attention backend. Since the 56fb5d8 build it works WITH DCP
# (DCP2/4/8 validated MTP-off upstream) and no longer needs fuse_allreduce_rms to
# run (fuse only adds ~8% and crashes on this host). NOT part of the adaptive sweep
# (backend toggles are A/B'd manually); usable via the attn_backend spec key in a
# --grid-file. "" means the image default, B12X_MLA_SPARSE.
SM120_BACKEND = "FLASHINFER_MLA_SPARSE_SM120"

# GLM-5.2's DSA indexer sparsity pattern. REQUIRED until upstream vLLM reads it from
# model config; without it the indexer mis-selects tokens and long context garbles.
INDEX_TOPK_PATTERN = ("FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"
                      "FSSSFSSSFSSSFSSSFSSSFSSSFSSS")

# DCP multiplies KV capacity ~linearly, so the safe single-request context length
# scales with it. These are conservative ceilings (below the measured KV pool at
# gpu_mem~0.95) so a config never boot-fails with "estimated max len < max_model_len".
MAXLEN_BY_DCP = {1: 256000, 2: 900000, 4: 1800000, 8: 3500000}

# Throughput bench scope (trimmed per request: cap concurrency at 8, 10s/cell).
# With concurrency capped at 8, max_num_seqs>8 can't be distinguished on throughput,
# which is why max_num_seqs is NOT a coordinate-descent axis (it is fixed at 32).
BENCH_CONC       = "1,2,4,8"
BENCH_CONTEXTS   = "0k,128k"
BENCH_DURATION   = 10
PREFILL_CONTEXTS = "8k,64k,128k"
PREFILL_DURATION = 10

# Fixed launch knobs shared by every config (correctness + the v12/v13 DCP fixes).
# B12X_MOE_FORCE_A16=1 is required for long-generation correctness; the two
# VLLM_DCP_* vars are the global-top-k / sharded-draft fixes (default-on in these
# images, set explicitly for clarity). max_num_seqs is fixed here (see note above).
FIXED_MAX_NUM_SEQS = 32


# =============================================================================
# Static-mode grid (used only by --mode static; adaptive mode ignores this).
# =============================================================================
# All entries default to the v13 backbone (make_experiment). fuse_allreduce_rms is
# intentionally absent — it faults on this 2-socket pod (no cross-socket P2P). See
# the adaptive() backbone note.
DEFAULT_GRID = [
    {"name": "dcp1_mtp3", "dcp": 1, "mtp": 3, "note": "256k fast path"},
    {"name": "dcp2_mtp3", "dcp": 2, "mtp": 3, "note": "baseline ~1M"},
    {"name": "dcp4_mtp3", "dcp": 4, "mtp": 3, "note": "~2M capacity"},
    {"name": "dcp2_mtp0", "dcp": 2, "mtp": 0},
    {"name": "dcp2_mtp2", "dcp": 2, "mtp": 2},
    {"name": "dcp2_mtp5", "dcp": 2, "mtp": 5},
    {"name": "dcp2_interleave1", "dcp": 2, "mtp": 3, "dcp_interleave1": True},
]


# =============================================================================
# Experiment specs
# =============================================================================
def make_experiment(spec):
    """Fill a partial spec dict with defaults to produce a complete experiment.

    Recognized keys (all optional except name): image, dcp, mtp, max_num_seqs,
    gpu_mem, max_model_len, a16, fuse_allreduce_rms, dcp_interleave1, pcie_backend
    (""/"cpp"/"b12x"), attn_backend (e.g. FLASHINFER_MLA_SPARSE_SM120), note.
    """
    dcp = int(spec.get("dcp", 2))
    image = spec.get("image", V13_IMAGE)   # v13 is the default backbone (v12 superseded)
    return {
        "name":               spec["name"],
        "image":              image,
        # AOT/compile cache is build-specific, so each image generation gets its own
        # dir (cache-v13-0629 is cold on first boot: expect a ~12-15 min compile).
        "cache_root":         spec.get("cache_root",
                                       os.path.join(GLM_DIR, "cache-v13-0629" if image == V13_IMAGE else "cache-v12")),
        "dcp":                dcp,
        "mtp":                int(spec.get("mtp", 3)),
        "max_num_seqs":       int(spec.get("max_num_seqs", FIXED_MAX_NUM_SEQS)),
        "gpu_mem":            float(spec.get("gpu_mem", 0.95)),
        "max_model_len":      int(spec.get("max_model_len", MAXLEN_BY_DCP.get(dcp, 256000))),
        "a16":                int(spec.get("a16", 1)),
        "fuse_allreduce_rms": bool(spec.get("fuse_allreduce_rms", False)),
        "dcp_interleave1":    bool(spec.get("dcp_interleave1", False)),
        "pcie_backend":       spec.get("pcie_backend", ""),
        "attn_backend":       spec.get("attn_backend", ""),
        "note":               spec.get("note", ""),
    }


def signature(e):
    """Stable identity of a config's *boot* — every field that changes the engine.

    Two specs with the same signature produce the same server, so a prior result can
    be reused instead of re-booting (this is what makes the sweep resumable and lets
    coordinate-descent phases share evaluations). Note: estonia_runs / bench_duration
    are deliberately excluded, so the confirmation phase must pass force=True to
    re-measure the same boot with more samples.
    """
    return "|".join(str(e[k]) for k in
                    ("image", "dcp", "mtp", "max_num_seqs", "gpu_mem", "max_model_len",
                     "a16", "fuse_allreduce_rms", "dcp_interleave1", "pcie_backend", "attn_backend"))


# =============================================================================
# Shell / Docker helpers
# =============================================================================
def run(cmd, timeout=None, env=None, log=None, cwd=None):
    """Run a command, never raising. Returns (returncode, combined_output).

    Output is also appended to `log` if given. A timeout returns rc=124 so callers
    can treat it like any other failure rather than crashing the sweep.
    """
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, timeout=timeout, env=env, cwd=cwd)
        out = (p.stdout or "") + (p.stderr or "")
        if log:
            open(log, "a").write(out)
        return p.returncode, out
    except subprocess.TimeoutExpired as ex:
        out = ((ex.stdout or "") + (ex.stderr or "")) if isinstance(ex.stdout, str) else ""
        if log:
            open(log, "a").write(out + f"\n[TIMEOUT {timeout}s]\n")
        return 124, out + f"\n[TIMEOUT {timeout}s]"


def teardown():
    """Remove the container and pause briefly for the GPUs/NCCL to release."""
    run(["docker", "rm", "-f", NAME], timeout=120)
    time.sleep(3)


def image_present(image):
    return bool(run(["docker", "images", "-q", image], timeout=60)[1].strip())


def build_env_and_args(e):
    """Translate an experiment into (env, EXTRA_ARGS) for launch.sh.

    launch.sh + entrypoint.sh apply this host's NCCL/b12x fixes and mount the caches;
    here we only set the per-config knobs. Returns the extra vllm-flag string too so
    it can be logged.
    """
    env = dict(os.environ)
    spec_tokens = 0 if e["mtp"] == 0 else e["mtp"]
    # CUDA graphs must be captured up to max_num_seqs * (1 + spec_tokens), because MTP
    # widens each scheduled sequence into (1 + draft) positions. Too small a capture
    # silently falls back to eager decode and collapses throughput at concurrency.
    cudagraph = e["max_num_seqs"] * (1 + spec_tokens)
    # JSON must be space-free: EXTRA_ARGS is word-split by launch.sh, and variable
    # expansion keeps the inner quotes literal, so this reaches vllm intact.
    chat_kwargs = '{"reasoning_effort":"high","clear_thinking":false}'

    extra = ["--max-cudagraph-capture-size", str(cudagraph),
             "--linear-backend", "auto",                 # b12x has no NVFP4 linear kernel for MTP eh_proj
             "--max-model-len", str(e["max_model_len"]),
             "--default-chat-template-kwargs", chat_kwargs]
    # Optional / experimental flags (the comm-tuning axis on this host):
    if e["dcp_interleave1"]:
        extra += ["--dcp-kv-cache-interleave-size", "1"]
    if e["fuse_allreduce_rms"]:
        extra += ["-cc.pass_config.fuse_allreduce_rms=True"]
    if e["attn_backend"]:                                # e.g. FLASHINFER_MLA_SPARSE_SM120 ("" = B12X default)
        extra += ["--attention-backend", e["attn_backend"]]

    env.update({
        "IMAGE": e["image"], "CACHE_ROOT": e["cache_root"], "NAME": NAME, "PORT": str(PORT),
        "TP_SIZE": "8", "DCP_SIZE": str(e["dcp"]),
        "GPU_MEMORY_UTILIZATION": str(e["gpu_mem"]), "MAX_NUM_SEQS": str(e["max_num_seqs"]),
        "NUM_SPECULATIVE_TOKENS": str(max(e["mtp"], 1)),     # used by serve only when MTP is enabled
        "GLM51_DISABLE_MTP": "1" if e["mtp"] == 0 else "0",
        "B12X_MOE_FORCE_A16": str(e["a16"]),                 # required for long-gen correctness
        "NCCL_MIN_NCHANNELS": "8",
        # 2026-07-03 P2P fix: NCCL (P2P-enabled via entrypoint default) beats the b12x
        # PCIe all-reduce, whose oneshot path collapses batch-1 decode (30 tok/s).
        "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_DCP_GLOBAL_TOPK": "1", "VLLM_DCP_SHARD_DRAFT": "1",   # v12/v13 DCP+MTP fixes
        "HF_OVERRIDES": json.dumps({"use_index_cache": True, "index_topk_pattern": INDEX_TOPK_PATTERN}),
        "EXTRA_ARGS": " ".join(extra),
    })
    if e["pcie_backend"]:                                    # "" leaves serve's default (cpp)
        env["VLLM_PCIE_ALLREDUCE_BACKEND"] = e["pcie_backend"]
    return env, " ".join(extra)


def wait_health(boot_timeout, log):
    """Poll /health until 200, the container dies, or timeout.

    Returns (ok, elapsed_seconds, reason). Boots are slow (cold compile + cudagraph
    capture), hence the long default timeout.
    """
    t0 = time.time()
    while time.time() - t0 < boot_timeout:
        if NAME not in run(["docker", "ps", "--format", "{{.Names}}"], timeout=30)[1].split():
            return False, time.time() - t0, "container_died"
        code = run(f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{PORT}/health", timeout=20)[1]
        if code.strip() == "200":
            return True, time.time() - t0, "healthy"
        time.sleep(10)
    return False, time.time() - t0, "boot_timeout"


def kv_cache_tokens():
    """Read the engine's reported GPU KV cache size (tokens) from the boot log."""
    out = run(["docker", "logs", NAME], timeout=60)[1]
    m = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", out)
    return int(m[-1].replace(",", "")) if m else 0


# =============================================================================
# Result-file parsers (read the bench's JSON, not its stdout)
# =============================================================================
def parse_estonia(path):
    """Pull the correctness summary from an Estonia profile run's JSON."""
    try:
        s = json.load(open(path)).get("selected_summary", {}) or {}
        sc = s.get("score_counts", {}) or {}
        return {"correct_rate": s.get("correct_rate", 0.0), "hit_max": s.get("hit_max_tokens", -1),
                "completed": s.get("completed", 0), "attempted": s.get("attempted", 0),
                "pass": int(sc.get("pass", sc.get("exact", 0))), "fail": int(sc.get("fail", 0)),
                "gen_tps": round(s.get("aggregate_gen_tok_s", 0.0), 1)}
    except Exception as ex:
        return {"correct_rate": 0.0, "hit_max": -1, "completed": 0, "attempted": 0,
                "pass": 0, "fail": 0, "gen_tps": 0.0, "_err": str(ex)}


def parse_bench(path):
    """Pull prefill tok/s, decode aggregate tok/s per (ctx, concurrency), and MTP
    acceptance length from a throughput-bench JSON. Missing cells become ""."""
    out = {}
    try:
        d = json.load(open(path))
        pf = d.get("prefill", {}) or {}
        out["prefill_8k"]   = round(pf.get("8192", {}).get("tok_per_sec", 0))
        out["prefill_64k"]  = round(pf.get("65536", {}).get("tok_per_sec", 0))
        out["prefill_128k"] = round(pf.get("131072", {}).get("tok_per_sec", 0))
        dec, accept = {}, []
        for r in d.get("results", []):
            dec[(r.get("context_tokens", 0), r.get("concurrency", 0))] = round(r.get("aggregate_tps", 0), 1)
            if r.get("server_spec_accept_length"):
                accept.append(r["server_spec_accept_length"])
        for c in (1, 2, 4, 8):
            out[f"dec_c{c}"] = dec.get((0, c), "")
        out["dec128k_c1"] = dec.get((131072, 1), "")
        out["dec128k_c8"] = dec.get((131072, 8), "")
        out["spec_accept_len"] = round(max(accept), 2) if accept else ""
    except Exception as ex:
        out["_bench_err"] = str(ex)
    return out


# =============================================================================
# CSV (one row per evaluated config; also the resume/dedup store)
# =============================================================================
CSV_FIELDS = ["ts", "phase", "name", "status", "note", "sig", "image_tag",
              "dcp", "mtp", "max_num_seqs", "gpu_mem", "max_model_len", "a16",
              "fuse_arrms", "interleave1", "pcie_backend", "attn_backend",
              "boot_s", "eval_s", "kv_cache_tokens",
              "estonia_correct_rate", "estonia_hit_max", "estonia_pass", "estonia_fail", "estonia_gen_tps",
              "prefill_8k", "prefill_64k", "prefill_128k",
              "dec_c1", "dec_c2", "dec_c4", "dec_c8", "dec128k_c1", "dec128k_c8", "spec_accept_len"]


def append_csv(csv_path, row):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def load_csv_cache(csv_path):
    """Load prior CSV rows as {signature: row} so a re-run reuses completed configs."""
    cache = {}
    if os.path.exists(csv_path):
        try:
            for row in csv.DictReader(open(csv_path)):
                if row.get("sig"):
                    cache[row["sig"]] = row
        except Exception:
            pass
    return cache


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# Core evaluation: one boot -> correctness gate -> bench -> one CSV row
# =============================================================================
def evaluate(e, ctx, phase, estonia_runs=None, bench_duration=None, force=False):
    """Evaluate one config and return a typed metrics dict (see normalize_metrics).

    Boots the engine, gates on the Estonia correctness profile, and only benches
    survivors. Always writes exactly one CSV row and always tears the container down.
    Any failure (pull/launch/boot/timeout/exception) is recorded as the row status
    and the sweep continues. force=True re-runs even if an identical signature is
    cached (used by confirmation to re-measure with more samples).
    """
    args, csv_path, outdir = ctx["args"], ctx["csv"], ctx["outdir"]
    estonia_runs = estonia_runs or args.estonia_runs
    bench_duration = bench_duration or args.bench_duration
    sig = signature(e)

    # Resume / cross-phase dedup: an identical boot was already measured.
    if not force and sig in ctx["cache"]:
        print(f"[{now()}] reuse cached: {e['name']} (sig match)")
        return normalize_metrics(ctx["cache"][sig])

    log = os.path.join(outdir, f"{e['name']}.log")
    est_json = os.path.join(outdir, f"{e['name']}.estonia.json")
    ben_json = os.path.join(outdir, f"{e['name']}.bench.json")
    t0 = time.time()
    row = {"ts": now(), "phase": phase, "name": e["name"], "note": e["note"], "sig": sig,
           "status": "started", "image_tag": e["image"].split(":")[-1],
           "dcp": e["dcp"], "mtp": e["mtp"], "max_num_seqs": e["max_num_seqs"],
           "gpu_mem": e["gpu_mem"], "max_model_len": e["max_model_len"], "a16": e["a16"],
           "fuse_arrms": int(e["fuse_allreduce_rms"]), "interleave1": int(e["dcp_interleave1"]),
           "pcie_backend": e["pcie_backend"] or "cpp", "attn_backend": e["attn_backend"] or "B12X_MLA_SPARSE"}

    def record(status):
        """Finalize this evaluation: stamp status + timing, persist, cache, return metrics."""
        row["status"] = status
        row["eval_s"] = int(time.time() - t0)
        append_csv(csv_path, row)
        ctx["cache"][sig] = row
        ctx["durations"].append(row["eval_s"])   # feeds the deadline estimator
        return normalize_metrics(row)

    print(f"\n[{now()}] === [{phase}] {e['name']} :: dcp{e['dcp']} mtp{e['mtp']} seqs{e['max_num_seqs']} "
          f"gpu{e['gpu_mem']} fuse{int(e['fuse_allreduce_rms'])} il{int(e['dcp_interleave1'])} "
          f"pcie={e['pcie_backend'] or 'cpp'} {e['attn_backend'] or ''} ===")
    try:
        teardown()  # clear any leftover container before launching

        if not image_present(e["image"]):
            print(f"[{now()}] pulling {e['image']} ...")
            if run(["docker", "pull", e["image"]], timeout=3600, log=log)[0] != 0 or not image_present(e["image"]):
                return record("image_pull_fail")

        env, extra = build_env_and_args(e)
        open(log, "a").write(f"\n===== {now()} {e['name']} =====\nEXTRA_ARGS: {extra}\n")
        if run(["bash", LAUNCH_SH], timeout=300, env=env, log=log)[0] != 0:
            return record("launch_fail")

        ok, boot_s, reason = wait_health(args.boot_timeout, log)
        row["boot_s"] = int(boot_s)
        if not ok:
            open(log, "a").write(f"\n[BOOT FAIL {reason}]\n{run(['docker','logs','--tail','40',NAME],timeout=60)[1]}\n")
            return record(f"boot_fail:{reason}")

        row["kv_cache_tokens"] = kv_cache_tokens()
        print(f"[{now()}] healthy {int(boot_s)}s, KV={row['kv_cache_tokens']:,}")

        # --- Estonia correctness gate (long-answer profile; a garbage detector) ---
        try:
            subprocess.run(["uv", "run", "python3", BENCH_PY, "--port", str(PORT), "--model", MODEL,
                            "--test-profile", "estonia", "--profile-concurrency", "2",
                            "--profile-runs", str(estonia_runs), "--max-tokens", "7000",
                            "--display-mode", "plain", "--output", est_json],
                           cwd=BENCH_DIR, capture_output=True, text=True, timeout=1500)
        except subprocess.TimeoutExpired:
            open(log, "a").write("\n[ESTONIA TIMEOUT]\n")
        est = parse_estonia(est_json)
        row.update({"estonia_correct_rate": est["correct_rate"], "estonia_hit_max": est["hit_max"],
                    "estonia_pass": est["pass"], "estonia_fail": est["fail"], "estonia_gen_tps": est["gen_tps"]})
        est_ok = est["correct_rate"] >= args.estonia_min_correct and est["completed"] > 0
        print(f"[{now()}] estonia correct_rate={est['correct_rate']} pass={est['pass']} "
              f"fail={est['fail']} hit_max={est['hit_max']}")
        if not est_ok and not args.run_bench_on_estonia_fail:
            return record("estonia_fail")   # skip the expensive bench on garbage output

        # --- trimmed throughput bench (decode ladder + standalone prefill) ---
        try:
            subprocess.run(["uv", "run", "python3", BENCH_PY, "--port", str(PORT), "--model", MODEL,
                            "--contexts", BENCH_CONTEXTS, "--concurrency", BENCH_CONC,
                            "--duration", str(bench_duration), "--prefill-contexts", PREFILL_CONTEXTS,
                            "--prefill-duration", str(PREFILL_DURATION),
                            "--display-mode", "plain", "--output", ben_json],
                           cwd=BENCH_DIR, capture_output=True, text=True, timeout=2400)
        except subprocess.TimeoutExpired:
            open(log, "a").write("\n[BENCH TIMEOUT]\n")
        row.update(parse_bench(ben_json))
        m = record("ok" if est_ok else "ok_estonia_fail")
        print(f"[{now()}] DONE {e['name']}: c1={row.get('dec_c1')} c8={row.get('dec_c8')} "
              f"pf8k={row.get('prefill_8k')} kv={row.get('kv_cache_tokens')} status={row['status']}")
        return m
    except Exception as ex:
        open(log, "a").write(f"\n[HARNESS EXCEPTION] {ex}\n")
        return record(f"error:{type(ex).__name__}")
    finally:
        teardown()


def normalize_metrics(row):
    """Coerce a CSV/row dict (values may be strings on resume) into a typed metrics
    dict the controller reasons about. `ok` = benched; `estonia_ok` = passed the gate."""
    def f(k, d=0.0):
        try:
            v = row.get(k, "")
            return float(v) if v not in ("", None) else d
        except Exception:
            return d
    return {
        "name": row.get("name", ""), "status": str(row.get("status", "")),
        "ok": str(row.get("status", "")).startswith("ok"),
        "estonia_ok": f("estonia_correct_rate") >= 0.5,
        "dcp": int(f("dcp", 2)), "mtp": int(f("mtp", 3)), "max_num_seqs": int(f("max_num_seqs", 32)),
        "gpu_mem": f("gpu_mem", 0.95),
        "fuse": int(f("fuse_arrms")), "interleave1": int(f("interleave1")),
        "pcie_backend": row.get("pcie_backend", "cpp"), "attn_backend": row.get("attn_backend", ""),
        "kv": f("kv_cache_tokens"), "accept": f("spec_accept_len"),
        "c1": f("dec_c1"), "c2": f("dec_c2"), "c4": f("dec_c4"), "c8": f("dec_c8"),
        "c128k_c1": f("dec128k_c1"), "c128k_c8": f("dec128k_c8"),
        "pf8k": f("prefill_8k"), "pf64k": f("prefill_64k"), "pf128k": f("prefill_128k"),
        "correct_rate": f("estonia_correct_rate"),
    }


# =============================================================================
# Multi-objective helpers
# =============================================================================
def pareto_front(results, objs=("c8", "pf8k", "kv")):
    """Non-dominated set over maximize(objs), among correct (Estonia-ok) benched
    configs. A config is dominated if another is >= on every objective and > on one.
    """
    pts = [m for m in results if m["ok"] and m["estonia_ok"]]
    front = []
    for a in pts:
        if not any(b is not a and all(b[o] >= a[o] for o in objs)
                   and any(b[o] > a[o] for o in objs) for b in pts):
            front.append(a)
    return front


def scalarize(m, weights, norms):
    """Weighted sum of metrics, each normalized by `norms` (so different-scale metrics
    like prefill ~2000 and decode ~80 combine fairly). Missing/zero norm -> 0 weight."""
    return sum(w * (m.get(k, 0) / norms[k]) for k, w in weights.items() if norms.get(k))


# =============================================================================
# Adaptive controller
# =============================================================================
def adaptive(ctx):
    """Run the three-phase adaptive search and write REPORT.md."""
    args = ctx["args"]
    results = []          # every metrics dict produced this run
    counter = {"n": 0}    # for unique, sortable config names

    def name_for(tag, e):
        counter["n"] += 1
        return (f"{counter['n']:03d}_{tag}_d{e['dcp']}m{e['mtp']}g{str(e['gpu_mem']).replace('.', '')}"
                f"{'F' if e['fuse_allreduce_rms'] else ''}{'I' if e['dcp_interleave1'] else ''}"
                f"{'B' if e['pcie_backend'] == 'b12x' else ''}{'X' if e['attn_backend'] else ''}")

    def ev(spec, phase, tag, **kw):
        """Name, evaluate, and record one spec; tag the result with its source spec."""
        e = make_experiment({**spec, "name": "tmp"})
        e["name"] = name_for(tag, e)
        m = evaluate(e, ctx, phase, **kw)
        m["_spec"] = spec          # let confirmation re-derive the exact config
        results.append(m)
        return m

    def time_left():
        return args.max_hours * 3600 - (time.time() - ctx["t_start"])

    def mean_eval():
        ds = ctx["durations"]
        return (sum(ds) / len(ds)) if ds else 14 * 60   # assume 14 min/eval until we have data

    def best_of(ms, weights):
        """Pick the highest-scoring correct config; fall back to `base` if none pass."""
        norms = {k: (max((m[k] for m in ms), default=1) or 1) for k in weights}
        ok = [m for m in ms if m["estonia_ok"] and m["ok"]]
        return max(ok or [base], key=lambda m: scalarize(m, weights, norms))

    # The "locked" recipe accumulates each phase's winner; cfg() materializes a full
    # spec from it (with the right max_model_len for the chosen DCP). Backbone is v13
    # "eldritch" 20260629 (canonical bugfix build). HOST CONSTRAINT: this is a 2-socket
    # pod with no working cross-socket GPU P2P, so fuse_allreduce_rms is forced OFF and
    # never tested — its fused *custom* all-reduce faults at init here (illegal memory
    # access). The attention backend (B12X vs FlashInfer SM120) is deliberately NOT an
    # axis here — backend/tech toggles are A/B'd manually outside the sweep; this
    # controller tunes hyperparameters only, all on B12X (the image default). pcie
    # backend is left at the image default (falls back to NCCL either way), so the
    # remaining comm-ish knob is the DCP KV-cache interleave.
    locked = {"image": V13_IMAGE, "dcp": 2, "mtp": 3, "max_num_seqs": FIXED_MAX_NUM_SEQS, "gpu_mem": 0.95,
              "fuse_allreduce_rms": False, "dcp_interleave1": False, "pcie_backend": "", "attn_backend": ""}

    def cfg(**ov):
        c = dict(locked); c.update(ov)
        c["max_model_len"] = MAXLEN_BY_DCP.get(c["dcp"], 256000)
        return c

    stop = ctx["stop"]
    print(f"[{now()}] ADAPTIVE start (v13 backbone). budget={args.max_hours}h, confirm-reserve={args.confirm_reserve_hours}h")

    # ---------- PHASE 1: greedy coordinate descent ----------
    base = ev(cfg(), "coord", "base")   # v13 DCP2 / MTP3 baseline

    if args.only_phase in (None, "coord", "all") and not stop["flag"]:
        # 1.1 DCP KV-cache interleave — the one comm-ish knob worth a look on this host.
        #     interleave=1 is v13's canonical default; test it vs off. Score on prefill +
        #     single-stream decode (the comm-sensitive metrics), equally weighted.
        ms = [base, ev(cfg(dcp_interleave1=True), "coord", "interleave")]
        locked["dcp_interleave1"] = bool(best_of(ms, {"pf8k": 0.5, "c1": 0.5})["interleave1"])
        print(f"[{now()}] LOCK interleave -> {locked['dcp_interleave1']}")

        # 1.2 MTP depth — spec decode mainly lifts low-concurrency decode; weight c1>c8.
        ms = [ev(cfg(mtp=mtp), "coord", "mtp") for mtp in (0, 2, 3, 5)]
        locked["mtp"] = best_of(ms, {"c1": 0.6, "c8": 0.4})["mtp"]
        print(f"[{now()}] LOCK mtp -> {locked['mtp']}")

        # 1.3 gpu_mem — KV scales with it; pick the highest that still boots and stays
        #     correct (Estonia exercises a ~130k-token prompt, so it doubles as a load
        #     check). Confirmation re-validates the chosen point.
        ms = [ev(cfg(gpu_mem=g), "coord", "gpu") for g in (0.90, 0.93, 0.95, 0.955, 0.96)]
        okm = [m for m in ms if m["estonia_ok"] and m["ok"] and m["kv"] > 0]
        if okm:
            locked["gpu_mem"] = max(okm, key=lambda m: m["kv"])["gpu_mem"]
        print(f"[{now()}] LOCK gpu_mem -> {locked['gpu_mem']}")

        # 1.4 DCP — kept as a Pareto axis (not collapsed): 1=fast/low-ctx, 2=~1M,
        #     4=max-ctx, at the tuned recipe (B12X at every DCP; the SM120 backend
        #     question is handled manually outside this sweep).
        for d in (1, 2, 4):
            ev(cfg(dcp=d), "coord", "dcp")

    # ---------- PHASE 2: budget-aware exploration ----------
    # Spend remaining time (minus the confirmation reserve) probing around the Pareto
    # front and a few random configs. Bounded by time, an iteration cap, and the stop
    # flag so it can never run away or block a graceful shutdown.
    if args.only_phase in (None, "all"):
        reserve = args.confirm_reserve_hours * 3600
        rng = random.Random(1234)
        explored = 0
        while (time_left() - reserve > mean_eval()) and explored < args.max_explore and not stop["flag"]:
            explored += 1
            front = pareto_front(results)
            if not front:
                break
            if rng.random() < 0.6:
                # local search: perturb one axis of a random Pareto member
                seed = rng.choice(front)
                ov = {"dcp": seed["dcp"], "mtp": seed["mtp"], "gpu_mem": seed["gpu_mem"],
                      "dcp_interleave1": bool(seed["interleave1"])}
                axis = rng.choice(["mtp", "gpu_mem", "dcp_interleave1"])
                if axis == "mtp":                ov["mtp"] = rng.choice([0, 2, 3, 5])
                elif axis == "gpu_mem":          ov["gpu_mem"] = rng.choice([0.93, 0.95, 0.955, 0.96])
                elif axis == "dcp_interleave1":  ov["dcp_interleave1"] = not ov["dcp_interleave1"]
            else:
                # random restart: a fresh point anywhere in the (host-viable) space
                ov = {"dcp": rng.choice([1, 2, 4]), "mtp": rng.choice([2, 3, 5]),
                      "gpu_mem": rng.choice([0.93, 0.95, 0.955]), "dcp_interleave1": rng.random() < 0.5}
            ev(cfg(**ov), "explore", "exp")

    # ---------- PHASE 3: confirmation ----------
    # Re-measure the Pareto front with more Estonia runs and a longer bench so the
    # final numbers are stable. Pick the per-objective leaders plus the rest of the
    # front (capped), and force=True so the cache doesn't short-circuit the re-run.
    front = pareto_front(results)
    picks, seen = [], set()
    for key in ("c8", "pf8k", "kv"):
        if front:
            w = max(front, key=lambda m: m[key])
            if w["name"] not in seen:
                picks.append(w); seen.add(w["name"])
    for m in front:
        if len(picks) >= 5:
            break
        if m["name"] not in seen:
            picks.append(m); seen.add(m["name"])

    if args.only_phase in (None, "all") and picks and not stop["flag"]:
        print(f"[{now()}] CONFIRM {len(picks)} Pareto configs (estonia_runs=5, bench 15s)")
        confirm_keys = ("image", "dcp", "mtp", "max_num_seqs", "gpu_mem",
                        "fuse_allreduce_rms", "dcp_interleave1", "pcie_backend", "attn_backend")
        for m in picks:
            sp = m.get("_spec")
            if not sp or time_left() < mean_eval() * 0.6:   # leave room to finish + report
                break
            ev(cfg(**{k: sp[k] for k in sp if k in confirm_keys}),
               "confirm", "cf", estonia_runs=5, bench_duration=15, force=True)

    write_report(ctx, results, locked)


def write_report(ctx, results, locked):
    """Write REPORT.md: the Pareto front table plus the per-use-case winners."""
    front = pareto_front(results)
    lines = [f"# GLM-5.2 adaptive sweep — {now()}", "",
             f"Tuned recipe (coordinate-descent winner): `{locked}`", "",
             f"Evaluated {len(results)} configs; {sum(1 for m in results if m['ok'])} ok, "
             f"{len(front)} on the Pareto front (maximize decode_c8 / prefill_8k / KV, Estonia-gated).", ""]

    cols = ["name", "dcp", "mtp", "gpu", "attn", "il", "c1", "c8", "pf8k", "kv", "accept", "correct"]

    def fmt(m):
        return {"name": m["name"], "dcp": m["dcp"], "mtp": m["mtp"], "gpu": m["gpu_mem"],
                "attn": "sm120" if m["attn_backend"] == SM120_BACKEND else "b12x",
                "il": m["interleave1"], "c1": m["c1"], "c8": m["c8"],
                "pf8k": m["pf8k"], "kv": int(m["kv"]), "accept": m["accept"], "correct": m["correct_rate"]}

    lines.append("## Pareto front")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for m in sorted(front, key=lambda m: -m["c8"]):
        r = fmt(m)
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")

    for label, key in (("Fastest decode (c8)", "c8"), ("Fastest prefill (8k)", "pf8k"), ("Max context (KV)", "kv")):
        if front:
            w = max(front, key=lambda m: m[key])
            attn = "sm120" if w["attn_backend"] == SM120_BACKEND else "b12x"
            lines.append(f"- **{label}**: `{w['name']}` — dcp{w['dcp']} mtp{w['mtp']} gpu{w['gpu_mem']} "
                         f"attn={attn} il{w['interleave1']} | c8={w['c8']} pf8k={w['pf8k']} "
                         f"kv={int(w['kv']):,} correct={w['correct_rate']}")

    path = os.path.join(ctx["outdir"], "REPORT.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"\n[{now()}] report -> {path}")
    print("\n".join(lines))


# =============================================================================
# Static mode
# =============================================================================
def static(ctx):
    """Evaluate DEFAULT_GRID (or --grid-file), skipping configs already in the CSV."""
    args = ctx["args"]
    grid = [make_experiment(s) for s in (json.load(open(args.grid_file)) if args.grid_file else DEFAULT_GRID)]
    if args.only:
        grid = [e for e in grid if e["name"] in set(args.only.split(","))]
    done = {r.get("name") for r in csv.DictReader(open(ctx["csv"]))} if os.path.exists(ctx["csv"]) else set()
    pending = [e for e in grid if e["name"] not in done]
    print(f"[{now()}] static: {len(grid)} total, {len(pending)} pending")
    if args.dry_run:
        for e in pending:
            _, extra = build_env_and_args(e)
            print(f"  - {e['name']:22s} dcp{e['dcp']} mtp{e['mtp']} seqs{e['max_num_seqs']} :: {extra}")
        return
    for e in pending:
        if ctx["stop"]["flag"] or (args.max_hours and (time.time() - ctx["t_start"]) / 3600 >= args.max_hours):
            break
        evaluate(e, ctx, "static")


# =============================================================================
# Offline self-test: run the controller against a synthetic model (no Docker).
# Validates control flow / phase progression / Pareto / report only — the numbers
# are made up and intentionally simple; it does NOT validate the real boot/bench.
# =============================================================================
def self_test(ctx):
    print("[self-test] mocking evaluate(); no Docker, no boots.")

    def mock_eval(e, ctx, phase, estonia_runs=None, bench_duration=None, force=False):
        dcp, mtp, g = e["dcp"], e["mtp"], e["gpu_mem"]
        kv = int(640000 * dcp * (g / 0.95))                       # KV ~ linear in DCP and gpu_mem
        c1 = (92 - (dcp - 1) * 4 + (8 + mtp if mtp else 0) * 0.6) \
            * (1.04 if e["fuse_allreduce_rms"] else 1.0) \
            * (0.99 if e["pcie_backend"] == "b12x" else 1.0) \
            * (1.06 if e["attn_backend"] and dcp == 1 else       # SM120 shines at DCP1,
               1.03 if e["attn_backend"] else 1.0)               # helps a bit at DCP>1
        c8 = c1 * (4.6 - (dcp - 1) * 0.3)
        pf = (4000 / dcp) * (1.06 if e["fuse_allreduce_rms"] else 1.0)
        boot_oom = (g >= 0.96 and dcp >= 4)                       # pretend the most aggressive combo OOMs
        a16_bad = (e["a16"] == 0)                                 # A16=0 -> incorrect output
        row = {"name": e["name"], "status": "boot_fail:oom" if boot_oom else "ok",
               "dcp": dcp, "mtp": mtp, "max_num_seqs": e["max_num_seqs"], "gpu_mem": g,
               "fuse_arrms": int(e["fuse_allreduce_rms"]), "interleave1": int(e["dcp_interleave1"]),
               "pcie_backend": e["pcie_backend"] or "cpp", "attn_backend": e["attn_backend"] or "",
               "kv_cache_tokens": 0 if boot_oom else kv,
               "estonia_correct_rate": 0.0 if a16_bad else 1.0,
               "dec_c1": 0 if boot_oom else round(c1, 1), "dec_c4": 0 if boot_oom else round(c1 * 3, 1),
               "dec_c8": 0 if boot_oom else round(c8, 1), "prefill_8k": 0 if boot_oom else round(pf),
               "prefill_64k": round(pf), "prefill_128k": round(pf), "spec_accept_len": 2.5 + 0.1 * mtp}
        ctx["durations"].append(30)
        return normalize_metrics(row)

    global evaluate
    real = evaluate
    evaluate = mock_eval
    ctx["args"].max_hours = 10.0            # time isn't the limiter under instant mocked evals
    ctx["args"].confirm_reserve_hours = 0.01
    ctx["args"].max_explore = 8             # the iteration cap is what ends exploration here
    try:
        adaptive(ctx)
    finally:
        evaluate = real
    print("[self-test] controller completed without error.")


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 serving-config sweep (static or adaptive).")
    ap.add_argument("--mode", choices=["static", "adaptive"], default="adaptive")
    ap.add_argument("--grid-file", help="static mode: JSON list of experiment specs (overrides DEFAULT_GRID)")
    ap.add_argument("--only", help="static mode: comma-separated experiment names to run")
    ap.add_argument("--dry-run", action="store_true", help="static mode: print the plan, launch nothing")
    ap.add_argument("--output-dir", default=os.path.join(GLM_DIR, "grid_results"))
    ap.add_argument("--csv", default=None, help="results CSV path (default: <output-dir>/results.csv)")
    ap.add_argument("--boot-timeout", type=int, default=1500, help="seconds to wait for /health")
    ap.add_argument("--bench-duration", type=int, default=BENCH_DURATION, help="seconds per decode cell")
    ap.add_argument("--estonia-runs", type=int, default=3)
    ap.add_argument("--estonia-min-correct", type=float, default=0.5,
                    help="Estonia is a garbage detector: this stack is non-deterministic at temp0 and the "
                         "model occasionally rambles past max_tokens, so a working config can score 2/3. "
                         "Gate at majority-correct (0.5); a broken config token-salads to ~0.")
    ap.add_argument("--run-bench-on-estonia-fail", action="store_true",
                    help="bench even if the Estonia gate fails (default: skip to save the boot's bench time)")
    ap.add_argument("--max-hours", type=float, default=14.0, help="hard wall-clock cap on the whole run")
    ap.add_argument("--confirm-reserve-hours", type=float, default=1.5,
                    help="adaptive: time reserved at the end for the confirmation phase")
    ap.add_argument("--max-explore", type=int, default=60,
                    help="adaptive: cap on exploration-phase evals (runaway backstop)")
    ap.add_argument("--only-phase", choices=["coord", "all"], default=None,
                    help="adaptive: 'coord' runs only coordinate descent (skip exploration)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the adaptive controller against mocked metrics (no Docker) and exit")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = args.csv or os.path.join(args.output_dir, "results.csv")
    ctx = {"args": args, "csv": csv_path, "outdir": args.output_dir,
           "cache": load_csv_cache(csv_path),   # prior results -> resume / dedup
           "durations": [],                      # per-eval seconds -> deadline estimator
           "t_start": time.time(),
           "stop": {"flag": False}}              # set by signal handler for graceful shutdown

    def handler(signum, frame):
        ctx["stop"]["flag"] = True
        print(f"\n[{now()}] interrupt — will stop after the current config (it still gets recorded)")
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    if args.self_test:
        return self_test(ctx)
    if args.mode == "static":
        return static(ctx)
    adaptive(ctx)
    print(f"\n[{now()}] adaptive sweep finished. csv={csv_path}")


if __name__ == "__main__":
    main()
