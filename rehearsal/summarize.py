"""Summarize a rehearsal output dir: python summarize.py DIR"""
import json
import sys

d = sys.argv[1]
log = [json.loads(l) for l in open(f"{d}/log.jsonl")]
ev = [json.loads(l) for l in open(f"{d}/eval.jsonl")]
checks = [c for r in log for c in r["checks"]]
honest = [c for c in checks if not c["attacker"]]
att = [c for c in checks if c["attacker"]]
first_caught = {}
for r in log:
    for c in r["checks"]:
        if c["attacker"] and not c["exact"]:
            first_caught.setdefault(c["peer"], r["round"])
out = {
    "rounds": len(log),
    "sec_per_round": sum(r["secs"] for r in log) / len(log),
    "honest_checks": len(honest), "honest_exact": sum(c["exact"] for c in honest),
    "honest_maxdiff": max((c["maxdiff"] for c in honest), default=None),
    "attacker_checks": len(att), "attacker_caught": sum(not c["exact"] for c in att),
    "attacker_first_caught_round": first_caught,
    "attacker_rounds_accepted": sum(int(p) in r["accepted"] for r in log for p in json.load(open(f"{d}/summary.json"))["cfg"]["ATTACK"]),
    "lp_gap_mean": sum(r["lp_gap_mean"] for r in log) / len(log), "lp_gap_max": max(r["lp_gap_max"] for r in log),
    "tis_clip": sum(r["tis_clip"] for r in log) / len(log),
    "truncated": sum(r["truncated"] for r in log) / len(log), "solved_frac": sum(r["solved_frac"] for r in log) / len(log),
    "reward_first5": sum(r["reward"] for r in log[:5]) / min(5, len(log)), "reward_last5": sum(r["reward"] for r in log[-5:]) / min(5, len(log)),
    "evals": ev,
}
print(json.dumps(out, indent=1))
