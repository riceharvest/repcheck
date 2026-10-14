"""Part C table: per scenario (mean over seeds) final clean acc, final/peak backdoor, bans, false rejections."""
import glob
import json
import re
import statistics as S

groups = {}
import sys
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/c_*.jsonl")):
    name = re.sub(r"_s\d+\.jsonl$", "", f.split("/")[-1])
    recs = [json.loads(l) for l in open(f)]
    fin = [r for r in recs if "final_eval" in r]
    if not fin:
        continue
    ev = [r for r in recs if "backdoor" in r]
    fin = fin[0]
    groups.setdefault(name, []).append({
        "eval": fin["final_eval"], "bd": fin["final_backdoor"], "bd_peak": max(r["backdoor"] for r in ev),
        "att_banned": len(set(fin["banned"]) & set(fin["attackers"])), "n_att": len(fin["attackers"]),
        "honest_banned": len(set(fin["banned"]) - set(fin["attackers"])),
        "honest_rej": fin["honest_rejected"], "checked": fin["checked"], "rejected": fin["rejected"],
        "withheld": fin["withheld"], "escal": fin["escalations"],
        "curve": [(r["round"], r["backdoor"], r["eval"]) for r in ev],
    })
print(f"{'scenario':24s} n  clean_acc  backdoor(final/peak)  attackers_banned honest_banned honest_rej checked rejected withheld escal")
out = {}
for name, rs in groups.items():
    m = lambda k: S.mean(r[k] for r in rs)
    print(f"{name:24s} {len(rs)}  {m('eval'):.3f}      {m('bd'):.3f}/{m('bd_peak'):.3f}           "
          f"{m('att_banned'):.1f}/{rs[0]['n_att']}            {m('honest_banned'):.1f}          {m('honest_rej'):.1f}      "
          f"{m('checked'):.0f}     {m('rejected'):.0f}      {m('withheld'):.0f}      {m('escal'):.0f}")
    out[name] = {"n": len(rs), **{k: m(k) for k in ["eval", "bd", "bd_peak", "att_banned", "honest_banned", "honest_rej", "checked", "rejected", "withheld", "escal"]},
                 "n_att": rs[0]["n_att"], "curve_s0": rs[0]["curve"],
                 "curve_mean": [(rs[0]["curve"][i][0], S.mean(r["curve"][i][1] for r in rs), S.mean(r["curve"][i][2] for r in rs))
                                for i in range(min(len(r["curve"]) for r in rs))]}
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "results/e3c_summary.json", "w"), indent=1)
