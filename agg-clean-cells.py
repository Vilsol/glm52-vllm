#!/usr/bin/env python3
"""Aggregate llm-inference-bench runs, keeping only cells with no external traffic.

A cell is CLEAN when the server never ran more requests than the bench asked for:
    max_running_reqs <= concurrency  AND  not underfilled
The tool itself only guards the low side (underfilled); it will happily report a
cell where someone else's request was in flight. See BENCHMARKS.md Phase 18 §9.
"""
import json, sys, glob
from statistics import mean, pstdev

def fmt(v):    return "-" if v is None else f"{v:,}" if isinstance(v,int) else f"{v:.1f}"
def ctxname(c): return "ctx0" if not c else (f"{c//1024}k" if c % 1024 == 0 else str(c))

cells = {}
for path in sorted(sys.argv[1:] or glob.glob("bench-c1-pass*.json")):
    try: d = json.load(open(path))
    except Exception as e:
        print(f"  !! {path}: {e}"); continue
    for r in d.get("results", []):
        conc = r.get("concurrency"); ctx = r.get("context_tokens") or 0
        mx = r.get("max_running_reqs"); av = r.get("avg_running_reqs")
        clean = (mx is not None and conc is not None and mx <= conc
                 and not r.get("underfilled"))
        cells.setdefault((ctx, conc), []).append({
            "run": path.split("/")[-1], "clean": clean,
            "tps": r.get("aggregate_tps"), "steps": r.get("server_steps_per_s"),
            "accept": r.get("server_accept_len_effective"), "avg": av, "max": mx,
        })

print(f"{'cell':>10} {'run':<34} {'ok':>3} {'avg/max':>9} {'tok/s':>8} {'steps/s':>8} {'accept':>7}")
for (ctx, conc), rows in sorted(cells.items()):
    for r in rows:
        print(f"{ctxname(ctx)+' c'+str(conc):>10} {r['run']:<34} "
              f"{'Y' if r['clean'] else 'no':>3} {str(r['avg'])+'/'+str(r['max']):>9} "
              f"{fmt(r['tps']):>8} {fmt(r['steps']):>8} {fmt(r['accept']):>7}")

print(f"\n{'CLEAN-ONLY AGGREGATE':>10}")
print(f"{'cell':>10} {'n':>3} {'tok/s mean':>11} {'sd':>7} {'steps/s mean':>13} {'sd':>7} {'accept':>7}")
for (ctx, conc), rows in sorted(cells.items()):
    ok = [r for r in rows if r["clean"] and r["tps"] is not None]
    if not ok:
        print(f"{ctxname(ctx)+' c'+str(conc):>10} {0:>3}   -- no clean sample --"); continue
    t = [r["tps"] for r in ok]; s = [r["steps"] for r in ok if r["steps"] is not None]
    a = [r["accept"] for r in ok if r["accept"] is not None]
    print(f"{ctxname(ctx)+' c'+str(conc):>10} {len(ok):>3} {mean(t):>11.1f} "
          f"{(pstdev(t) if len(t)>1 else 0):>7.1f} {(mean(s) if s else 0):>13.1f} "
          f"{(pstdev(s) if len(s)>1 else 0):>7.1f} {(mean(a) if a else 0):>7.2f}")
