#!/usr/bin/env python3
"""Greedy byte-identical check for VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE.

Speculative decoding must be LOSSLESS: with temperature=0 the accepted output
has to be byte-identical to what the same model produces without speculation.
The v19 in-tree comment warns that routing the MTP q_len=4 verify batch through
the sparse DECODE kernel (the flag) breaks the causal masking between draft rows
-- which would corrupt the verifier's logits *silently*. festr reports it is
lossless and v20 enables it unconditionally, but that is on the v20 kernel.
This script is how we verify it on OUR v19 build before trusting it.

Usage:
    ./spec-extend-lossless-check.py capture <label>   # run prompts, save outputs
    ./spec-extend-lossless-check.py compare <a> <b>   # diff two captures

Procedure (each capture needs its own boot, so run in this order):
    1. boot with VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=0   -> capture flagoff
    2. boot with VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1   -> capture flagon
    3. compare flagoff flagon      -> ANY diff means the flag is NOT lossless

Everything is greedy (temperature=0, top_p=1.0 is intentional here: we want the
raw argmax path, not the 0.95 nucleus the server defaults to -- a truncated
distribution could mask a logit difference). seed is pinned too.
"""
import json
import sys
import urllib.request
from pathlib import Path

PORT = 8443
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
OUTDIR = Path("/root/glm52-vllm/lossless-captures")

# Prompts chosen to force LONG deterministic generations: the more tokens, the
# more MTP verify steps, and a logit corruption only has to bite once to show up.
# Mixed domains because the failure mode reported in Discord (CJK drift) was
# language-dependent.
PROMPTS = [
    ("count", "Count from 1 to 200, separated by commas. Output only the numbers."),
    ("code", "Write a complete Python implementation of a red-black tree with "
             "insert, delete, and search. Include docstrings. No explanation."),
    ("prose", "Explain how a four-stroke internal combustion engine works, in "
              "exactly 12 numbered steps, one paragraph each."),
    ("recall", "List the first 60 prime numbers, then the first 40 Fibonacci "
               "numbers, each on its own line with an index."),
    ("cjk", "Explain the theory of relativity in Simplified Chinese, then "
            "translate your explanation to English. Be thorough."),
]

MAX_TOKENS = 2048


def one(prompt: str) -> dict:
    body = {
        "model": "GLM-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 12345,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode())
    msg = d["choices"][0]["message"]
    # GLM-5.2 is a reasoning model: a prompt can burn the whole token budget in
    # thinking and return content=None. That is where most of the generated
    # tokens live, so it must be part of the comparison -- comparing only
    # `content` would silently skip 2048 tokens of decode.
    # NB: this server names the field "reasoning", NOT "reasoning_content".
    return {
        "text": msg.get("content"),
        "reasoning": msg.get("reasoning"),
        "finish_reason": d["choices"][0].get("finish_reason"),
        "usage": d.get("usage", {}),
    }


def capture(label: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, prompt in PROMPTS:
        print(f"  {name} ...", flush=True, end="")
        r = one(prompt)
        out[name] = r
        print(f" {r['usage'].get('completion_tokens', '?')} tok")
    path = OUTDIR / f"{label}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"saved {path}")


def compare(a: str, b: str) -> int:
    da = json.loads((OUTDIR / f"{a}.json").read_text())
    db = json.loads((OUTDIR / f"{b}.json").read_text())
    bad = 0
    for name, _ in PROMPTS:
        # compare content and reasoning as one stream; either may be None
        ta = (da.get(name, {}).get("text") or "") + (da.get(name, {}).get("reasoning") or "")
        tb = (db.get(name, {}).get("text") or "") + (db.get(name, {}).get("reasoning") or "")
        if not ta or not tb:
            print(f"{name:8s} EMPTY in one capture (old capture without reasoning?)")
            bad += 1
            continue
        if ta == tb:
            print(f"{name:8s} IDENTICAL ({len(ta)} chars)")
            continue
        bad += 1
        # first divergence point tells us how deep the generation got before
        # the kernels disagreed -- an early split is a much stronger signal.
        i = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
                 min(len(ta), len(tb)))
        print(f"{name:8s} DIFFERS at char {i} of {len(ta)}/{len(tb)}")
        print(f"    {a}: ...{ta[max(0,i-60):i+60]!r}")
        print(f"    {b}: ...{tb[max(0,i-60):i+60]!r}")
    print()
    if bad:
        print(f"RESULT: NOT LOSSLESS — {bad}/{len(PROMPTS)} prompts diverged")
    else:
        print(f"RESULT: LOSSLESS — all {len(PROMPTS)} prompts byte-identical")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "capture":
        capture(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(2)
