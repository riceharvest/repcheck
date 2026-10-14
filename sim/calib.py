"""Per-class mismatch vs the fp32 trainer: KL(q||p) on sampled answer tokens + MoE routing flips."""
import json
import torch
from core import *

base = torch.load("base.pt")
lora = init_lora(base)
# give the adapter a non-trivial value so LoRA numerics are exercised too
g = torch.Generator().manual_seed(5)
for k in lora:
    lora[k][1] = torch.randn(lora[k][1].shape, generator=g) * 0.01
gen = torch.Generator().manual_seed(3)
a, b = make_prompts(1024, gen)
prompts = prompt_tokens(a, b)
ref = make_class_base(base, "fp32")
res = {}
for cls in CLASSES:
    bc, dt = make_class_base(base, cls)
    out = sample(bc, dt, lora, prompts, seed_of("calib", cls))
    kl, flips, ntok = 0.0, 0.0, 0
    for L in sorted(set(out["plen"])):
        idx = [i for i, pl in enumerate(out["plen"]) if pl == L]
        toks = torch.stack([out["tokens"][i] for i in idx])
        with torch.no_grad():
            lp, r_ref = forward(ref[0], lora, toks[:, :-1], ref[1], return_routes=True)
            _, r_c = forward(bc, lora, toks[:, :-1], dt, return_routes=True)
        lp = lp[:, L - 1:].log_softmax(-1)
        lq = torch.stack([out["logq"][i] for i in idx])
        valid = torch.zeros(lq.shape[:2], dtype=torch.bool)
        for j, i in enumerate(idx):
            valid[j, : answer_valid_len(out["tokens"][i][L:].tolist())] = True
        q = lq.exp()
        kl_full = (q * (lq - lp)).sum(-1)  # exact KL(q||p) per position
        kl += kl_full[valid].sum().item()
        ntok += valid.sum().item()
        for l in range(NL):
            flips += (r_ref[l] != r_c[l]).any(-1).float().mean().item() * len(idx) / NL
    acc = evaluate(bc, dt, lora, a[:512], b[:512], seed=1)
    res[cls] = {"kl": kl / ntok, "route_flip": flips / len(prompts), "acc": acc}
    print(cls, {k: f"{v:.2e}" if k == "kl" else round(v, 4) for k, v in res[cls].items()}, flush=True)
json.dump(res, open("results/calib.json", "w"), indent=1)
