"""E1: one fp32 trainer, many samplers in mixed hardware classes, async staleness.

Mirrors the Modal setup: samplers use their own class + a stale adapter version,
trainer applies pg / mis / sc / sc_mis (+ optional top-p mask replay).
"""
import argparse
import json
import random
import time

import torch
from core import *

ap = argparse.ArgumentParser()
ap.add_argument("--classes", default="fp32")
ap.add_argument("--method", default="pg")
ap.add_argument("--top_p", type=float, default=1.0)
ap.add_argument("--stale", type=int, default=0, help="max staleness (uniform 0..K)")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--prompts", type=int, default=32)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--noise", type=float, default=0.0, help="extra sampler weight noise")
ap.add_argument("--train_A", action="store_true")
ap.add_argument("--out", required=True)
args = ap.parse_args()

random.seed(args.seed)
base = torch.load("base.pt")
trainer_base, trainer_dt = make_class_base(base, "fp32")
classes = args.classes.split(",")
cls_bases = {}
for c in classes:
    bc, dt = make_class_base(base, c)
    if args.noise > 0:
        bc = weight_noise_base(bc, args.noise, seed_of("noise", c, args.seed))
    cls_bases[c] = (bc, dt)

lora = init_lora(base)
params = lora_params(lora, train_A=args.train_A)
opt = torch.optim.Adam(params, lr=args.lr)
history = [clone_lora(lora)]
gen = torch.Generator().manual_seed(100 + args.seed)
eg = torch.Generator().manual_seed(7)
ea, eb = make_prompts(512, eg)
log = open(args.out, "w")
t0 = time.time()
for step in range(args.steps + 1):
    if step % 10 == 0:
        acc = evaluate(trainer_base, trainer_dt, {k: [A.detach(), B.detach()] for k, (A, B) in lora.items()}, ea, eb, seed=step)
    a, b = make_prompts(args.prompts, gen)
    # assign groups to (class, staleness)
    jobs = {}
    for gi in range(args.prompts):
        c = classes[gi % len(classes)]
        k = random.randint(0, min(args.stale, len(history) - 1))
        jobs.setdefault((c, k), []).append(gi)
    batch = {"tokens": [], "logq": [], "mask": [], "plen": []}
    rewards, meta = [], []
    for (c, k), gis in jobs.items():
        ver = history[-1 - k]
        pa = [a[g] for g in gis for _ in range(args.G)]
        pb = [b[g] for g in gis for _ in range(args.G)]
        out = sample(cls_bases[c][0], cls_bases[c][1], ver, prompt_tokens(pa, pb), seed_of(args.seed, step, c, k), top_p=args.top_p)
        for i in range(len(pa)):
            batch["tokens"].append(out["tokens"][i])
            batch["logq"].append(out["logq"][i])
            batch["mask"].append(out["mask"][i])
            batch["plen"].append(out["plen"][i])
            rewards.append(reward(pa[i], pb[i], out["tokens"][i][out["plen"][i]:].tolist()))
    adv, mixed = group_advantages(rewards, args.G)
    loss, st = rl_loss(trainer_base, trainer_dt, lora, batch, adv, args.method, top_p_replay=args.top_p < 1.0)
    opt.zero_grad()
    loss.backward()
    gn = torch.sqrt(sum((p.grad ** 2).sum() for p in params)).item()
    opt.step()
    history.append(clone_lora({k: [A.detach(), B.detach()] for k, (A, B) in lora.items()}))
    history = history[-(args.stale + 2):]
    rec = {"step": step, "reward": sum(rewards) / len(rewards), "mixed": mixed, "gn": gn, **st}
    if step % 10 == 0:
        rec["eval"] = acc
        print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}), f"{time.time()-t0:.0f}s", flush=True)
    log.write(json.dumps(rec) + "\n")
log.close()
