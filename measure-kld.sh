#!/usr/bin/env bash
###############################################################################
# Measure the output-distribution divergence introduced by compressed PCIe DMA
# (F8_DMA=ring / i8_ring / ...) RELATIVE TO the lossless path (F8_DMA=0).
#
# WHY THIS IS MEASURABLE HERE, WHEN "REAL" KLD IS NOT:
#   The community KLD table (lavd_48722) is measured against a BF16 reference.
#   We cannot reproduce that: BF16 GLM-5.2 is ~1.5 TB and will not fit in our
#   768 GB of VRAM. But that is not the question we need answered.
#
#   F8_DMA=0 and F8_DMA=ring run IDENTICAL weights with IDENTICAL quantization.
#   The ONLY difference is the wire format of the TP all-reduce, and the probe
#   labels the uncompressed path numeric_contract="lossless-only". So F8_DMA=0
#   is a valid LOCAL reference, and the delta we measure is exactly the quality
#   cost that compression ADDS on top of NVFP4 A16.
#
#   Our number is therefore NOT comparable in absolute terms to the published
#   table (0.059940 etc). It is an incremental divergence, not a vs-BF16 KLD.
#   Report it as such or it will be misread.
#
# *** THE PROMPTS MUST BE LONG. ***
#   The b12x DMA path only engages at payloads >= PCIE_DMA_MIN_BYTES (24 MB),
#   which is 2048 rows == 2048 tokens in a batched chunk. A short prompt routes
#   through PyNCCL, both configs are then bit-identical, and this harness would
#   report ~0 divergence and "prove" compression is free. That would be an
#   artifact of the prompt length, not a result. We use ~8k-token prompts so a
#   full 8192-row / 96 MiB chunk is exercised.
#
# USAGE:
#   ./measure-kld.sh capture <label>    # capture logprobs from the running server
#   ./measure-kld.sh compare <ref> <test>
###############################################################################
set -uo pipefail
OUT=/root/glm52-vllm/kld
PORT="${PORT:-8443}"
BASE="http://127.0.0.1:$PORT"
mkdir -p "$OUT"

build_corpus() {
  # Deterministic mixed corpus assembled from files already on disk, so the
  # measurement is reproducible without network access. Mix of technical prose,
  # shell code and natural conversation, since divergence can be content
  # dependent (code has sharper distributions than chat).
  python3 - <<'PY'
import json, os, glob
srcs = []
def add(path, n):
    try: srcs.append(open(path, errors='ignore').read()[:n])
    except Exception: pass
add('/root/glm52-vllm/vllm-v20-r20.sh', 60000)                 # shell / config
for f in sorted(glob.glob('/tmp/claude-0/*/*/scratchpad/dc2/2026-07-29.txt'))[:1]:
    add(f, 60000)                                              # natural chat
add('/root/glm52-vllm/r20-inspect/serve-glm52-v16.sh', 60000)  # more code
blob = "\n\n".join(srcs)
# ~4 chars/token heuristic -> ~8k tokens per prompt, 4 prompts
per = 32000
prompts = [blob[i*per:(i+1)*per] for i in range(4)]
prompts = [p for p in prompts if len(p) > 20000]
json.dump(prompts, open('/root/glm52-vllm/kld/corpus.json','w'))
print(f"corpus: {len(prompts)} prompts, ~{per//4} tokens each")
PY
}

capture() {
  local label="$1"
  [[ -f "$OUT/corpus.json" ]] || build_corpus
  python3 - "$label" <<'PY'
import json, sys, urllib.request
label = sys.argv[1]
prompts = json.load(open('/root/glm52-vllm/kld/corpus.json'))
out = []
for i, p in enumerate(prompts):
    body = json.dumps({
        "model": "GLM-5.2", "prompt": p,
        "max_tokens": 1, "temperature": 0.0, "seed": 0,
        "prompt_logprobs": 20,          # top-20 distribution at every position
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:8443/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    pl = d["choices"][0].get("prompt_logprobs") or []
    # keep only {token_id: logprob}; drop decoded text to keep files small
    slim = [None if e is None else {k: v["logprob"] for k, v in e.items()} for e in pl]
    out.append(slim)
    print(f"  prompt {i}: {len(slim)} positions")
json.dump(out, open(f'/root/glm52-vllm/kld/logprobs-{label}.json','w'))
print(f"saved kld/logprobs-{label}.json")
PY
}

compare() {
  python3 - "$1" "$2" <<'PY'
import json, sys, math
ref_l, test_l = sys.argv[1], sys.argv[2]
R = json.load(open(f'/root/glm52-vllm/kld/logprobs-{ref_l}.json'))
T = json.load(open(f'/root/glm52-vllm/kld/logprobs-{test_l}.json'))
kls, top1_same, n, absdelta = [], 0, 0, []
for pr, pt in zip(R, T):
    for a, b in zip(pr, pt):
        if not a or not b: continue
        # renormalise the reference top-k over tokens present in BOTH, then
        # KL(ref || test). Truncated-support KL: standard practical estimator.
        common = set(a) & set(b)
        if len(common) < 2: continue
        za = math.log(sum(math.exp(a[k]) for k in common))
        zb = math.log(sum(math.exp(b[k]) for k in common))
        kl = 0.0
        for k in common:
            pa = math.exp(a[k] - za)
            kl += pa * ((a[k] - za) - (b[k] - zb))
        kls.append(max(kl, 0.0))
        ta = max(a, key=a.get); tb = max(b, key=b.get)
        top1_same += (ta == tb); n += 1
        absdelta.append(abs(a[ta] - b.get(ta, b[tb])))
kls.sort()
mean = sum(kls)/len(kls)
print(f"positions compared : {n}")
print(f"mean KL(ref||test) : {mean:.6f}")
print(f"median KL          : {kls[len(kls)//2]:.6f}")
print(f"p99 KL             : {kls[int(len(kls)*0.99)]:.6f}")
print(f"max KL             : {kls[-1]:.6f}")
print(f"top-1 agreement    : {100*top1_same/n:.3f}%  ({n-top1_same} tokens differ)")
print(f"mean |dlogprob| of ref top-1 : {sum(absdelta)/len(absdelta):.6f}")
print()
print("NOTE: incremental divergence vs the LOSSLESS local path, NOT a vs-BF16")
print("      KLD. Do not compare directly to the published 0.0599-style table.")
PY
}


generate() {
  # Greedy generation from the SAME long prompts. Teacher-forced logprobs bound
  # per-token divergence; this bounds END-TO-END divergence, because one flipped
  # token redirects every token after it. Prompts must stay >=2048 tokens so the
  # prefill actually exercises the DMA path (decode payloads are ~12-48 KB and
  # always take PyNCCL regardless of config).
  local label="$1"
  [[ -f "$OUT/corpus.json" ]] || build_corpus
  python3 - "$label" <<'PYG'
import json, sys, urllib.request
label = sys.argv[1]
prompts = json.load(open('/root/glm52-vllm/kld/corpus.json'))
out=[]
for i,p in enumerate(prompts):
    body=json.dumps({"model":"GLM-5.2","prompt":p[:16000],
                     "max_tokens":256,"temperature":0.0,"seed":0,
                     "logprobs":1}).encode()
    req=urllib.request.Request("http://127.0.0.1:8443/v1/completions",
        data=body, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r: d=json.load(r)
    ch=d["choices"][0]
    lp=ch.get("logprobs") or {}
    out.append({"text":ch.get("text",""), "tokens":lp.get("tokens",[])})
    print(f"  prompt {i}: {len(lp.get('tokens',[]))} tokens generated")
json.dump(out, open(f'/root/glm52-vllm/kld/gen-{label}.json','w'))
print(f"saved kld/gen-{label}.json")
PYG
}

gencompare() {
  python3 - "$1" "$2" <<'PYG'
import json, sys
a=json.load(open(f'/root/glm52-vllm/kld/gen-{sys.argv[1]}.json'))
b=json.load(open(f'/root/glm52-vllm/kld/gen-{sys.argv[2]}.json'))
print(f"{'prompt':>7}{'gen len':>9}{'first divergence':>18}{'identical tokens':>18}")
for i,(x,y) in enumerate(zip(a,b)):
    tx,ty=x["tokens"],y["tokens"]
    n=min(len(tx),len(ty))
    first=next((j for j in range(n) if tx[j]!=ty[j]), None)
    same=sum(1 for j in range(n) if tx[j]==ty[j])
    fd = "identical" if first is None else f"token {first}"
    print(f"{i:>7}{n:>9}{fd:>18}{f'{100*same/n:.1f}%':>18}")
PYG
}

case "${1:-}" in
  capture) capture "${2:?need a label}" ;;
  compare) compare "${2:?need ref label}" "${3:?need test label}" ;;
  corpus)  build_corpus ;;
  generate) generate "${2:?need a label}" ;;
  gencompare) gencompare "${2:?}" "${3:?}" ;;
  *) echo "usage: $0 {corpus|capture <l>|compare <ref> <test>|generate <l>|gencompare <ref> <test>}" >&2; exit 1 ;;
esac
