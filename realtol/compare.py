"""Cross-platform tolerance: replay a peer's decodes from one platform on another (same format).

python realtol/compare.py <peer .pt> <local device> [fmt]
  - token/log-prob replay of the peer's decodes on the local device
  - fixed-sequence ref log-prob deviation vs the local run's .pt (if given via LOCAL_PT)
  - LoRA delta relative error vs the local run (if LOCAL_PT has deltas)
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import realtol as R  # noqa: E402

peer_pt, dev = sys.argv[1], sys.argv[2]
P = torch.load(peer_pt, weights_only=False)
model_tag, fmt = P["tag"].split("|")
model_id = {"Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct", "OLMoE-1B-7B-0125-Instruct": "allenai/OLMoE-1B-7B-0125-Instruct"}[model_tag]
fmt_local = sys.argv[3] if len(sys.argv) > 3 else fmt
R.N_PROMPTS = len(P["dec"])
R.MAX_NEW = P["dec"][0]["tokens"].shape[1]
out = {"peer": peer_pt, "local_device": dev, "fmt_peer": fmt, "fmt_local": fmt_local}
m = R.load_model(model_id, fmt_local, dev)
out["replay_peer_decodes"] = R.replay(m, P["dec"], dev, P["tag"], R.N_SAMPLES)
local_pt = os.environ.get("LOCAL_PT")
if local_pt:
    L = torch.load(local_pt, weights_only=False)
    peer_dev = [k for k in P["fixed"] if k.startswith("cuda") or k.startswith("xpu") or k == "cpu"][0]
    loc_dev = list(L["fixed"])[0]
    a, b = P["fixed"][peer_dev]["b8"], L["fixed"][loc_dev]["b8"]
    d = torch.cat([(x["ref_lp"] - y["ref_lp"]).abs() for x, y in zip(a, b)])
    out["fixed_ref_lp_dev"] = {"p50": float(d.median()), "p99": float(torch.quantile(d.double(), 0.99)), "max": float(d.max())}
    pk = [k for k in P["deltas"] if k.endswith("/mb8")]
    lk = [k for k in L["deltas"] if k.endswith("/mb8")]
    if pk and lk:
        out["delta_rel_cross"] = R.rel(P["deltas"][pk[0]], L["deltas"][lk[0]])
print(json.dumps(out, indent=1))
os.makedirs("results/realtol", exist_ok=True)
name = os.path.basename(peer_pt).replace(".pt", "")
json.dump(out, open(f"results/realtol/cross_{name}_on_{dev.replace(':', '')}_{fmt_local}.json", "w"), indent=1)
