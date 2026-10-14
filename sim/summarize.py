"""Summarise jsonl runs: group by name minus _s<seed>, report eval curve, peak, final, KL growth."""
import glob
import json
import re
import statistics as S
import sys

pat = sys.argv[1] if len(sys.argv) > 1 else "results/*.jsonl"
groups = {}
for f in sorted(glob.glob(pat)):
    name = re.sub(r"_s\d+\.jsonl$", "", f.split("/")[-1])
    recs = [json.loads(l) for l in open(f)]
    ev = [(r.get("steq", r.get("step")), r["eval"]) for r in recs if "eval" in r]
    kl = [r["kl"] for r in recs if "kl" in r]
    ex = [r["extreme"] for r in recs if "extreme" in r]
    fin = [r for r in recs if "final_per_class" in r]
    if not ev:
        continue
    last = [e for _, e in ev[-5:]]
    n = len(kl)
    groups.setdefault(name, []).append({
        "final": S.mean(last), "peak": max(e for _, e in ev),
        "kl_early": S.mean(kl[n // 10: n // 5]) if n > 10 else float("nan"),
        "kl_late": S.mean(kl[-max(1, n // 10):]),
        "extreme_late": S.mean(ex[-max(1, n // 10):]) if ex else float("nan"),
        "steps": ev[-1][0],
        "curve": [round(e, 3) for s, e in ev if s % 100 == 0],
        "final_per_class": fin[0]["final_per_class"] if fin else None,
    })
print(f"{'run':38s} {'n':>2s} {'final':>13s} {'peak':>6s} {'KLearly':>8s} {'KLlate':>8s} {'xtrm%':>6s}  curve(seed0, every 100)")
for name, rs in groups.items():
    fm = S.mean(r["final"] for r in rs)
    fs = S.pstdev(r["final"] for r in rs) if len(rs) > 1 else 0
    print(f"{name:38s} {len(rs):2d} {fm:6.3f}±{fs:5.3f} {S.mean(r['peak'] for r in rs):6.3f} "
          f"{S.mean(r['kl_early'] for r in rs):8.4f} {S.mean(r['kl_late'] for r in rs):8.4f} "
          f"{100*S.mean(r['extreme_late'] for r in rs):6.2f}  {rs[0]['curve']}")
    rs = [r for r in rs if r["final_per_class"]] or rs
    if rs[0]["final_per_class"]:
        pc = {c: round(S.mean(r["final_per_class"][c] for r in rs), 3) for c in rs[0]["final_per_class"]}
        print(f"{'':38s}    per-class final: {pc}")
