"""E2: no trainer cluster. Every peer trains the adapter on its OWN rollouts with its OWN quantized
model, then adapters are averaged (FedAvg / DiLoCo-style). Peers churn (go offline) and return with
stale globals.

FFA-LoRA (A frozen from a shared seed, only B averaged) vs standard LoRA (A and B averaged separately).
"""
import argparse
import json
import random
import time

import torch
from core import *

ap = argparse.ArgumentParser()
ap.add_argument("--classes", default="fp32,bf16,fp16,int8,int4,int3")
ap.add_argument("--peers", type=int, default=12)
ap.add_argument("--online", type=float, default=0.7, help="P(peer online in a round)")
ap.add_argument("--lag", type=int, default=2, help="max rounds a peer's starting global can be behind")
ap.add_argument("--H", type=int, default=4, help="local steps per round")
ap.add_argument("--prompts", type=int, default=4, help="prompts per local step per peer")
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--rounds", type=int, default=150)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--outer_lr", type=float, default=1.0)
ap.add_argument("--method", default="sc_mis")
ap.add_argument("--train_A", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

random.seed(args.seed)
base = torch.load("base.pt")
classes = args.classes.split(",")
peer_cls = [classes[i % len(classes)] for i in range(args.peers)]
cls_bases = {c: make_class_base(base, c) for c in set(peer_cls)}
ref_base, ref_dt = make_class_base(base, "fp32")

glob = init_lora(base)  # global adapter (A shared/frozen unless train_A)
history = [clone_lora(glob)]
gen = torch.Generator().manual_seed(200 + args.seed)
eg = torch.Generator().manual_seed(7)
ea, eb = make_prompts(512, eg)
log = open(args.out, "w")
t0 = time.time()


def local_train(pid, start, rnd):
    c = peer_cls[pid]
    bc, dt = cls_bases[c]
    loc = clone_lora(start)
    ps = lora_params(loc, train_A=args.train_A)
    opt = torch.optim.Adam(ps, lr=args.lr)
    rs, kls = [], []
    for h in range(args.H):
        a, b = make_prompts(args.prompts, gen)
        pa = [x for x in a for _ in range(args.G)]
        pb = [x for x in b for _ in range(args.G)]
        det = {k: [A.detach(), B.detach()] for k, (A, B) in loc.items()}
        out = sample(bc, dt, det, prompt_tokens(pa, pb), seed_of(args.seed, rnd, pid, h))
        rew = [reward(pa[i], pb[i], out["tokens"][i][out["plen"][i]:].tolist()) for i in range(len(pa))]
        adv, _ = group_advantages(rew, args.G)
        loss, st = rl_loss(bc, dt, loc, out, adv, args.method)
        opt.zero_grad()
        loss.backward()
        opt.step()
        rs += rew
        kls.append(st["kl"])
    delta = {k: [(loc[k][0].detach() - start[k][0]), (loc[k][1].detach() - start[k][1])] for k in loc}
    return delta, sum(rs) / len(rs), sum(kls) / len(kls)


for rnd in range(args.rounds + 1):
    if rnd % 5 == 0:
        acc = evaluate(ref_base, ref_dt, glob, ea, eb, seed=rnd)
    active = [p for p in range(args.peers) if random.random() < args.online] or [0]
    deltas, rews, kls = [], [], []
    for p in active:
        lag = random.randint(0, min(args.lag, len(history) - 1))
        d, r, k = local_train(p, history[-1 - lag], rnd)
        deltas.append(d)
        rews.append(r)
        kls.append(k)
    new = clone_lora(glob)
    for k in new:
        dA = torch.stack([d[k][0] for d in deltas]).mean(0)
        dB = torch.stack([d[k][1] for d in deltas]).mean(0)
        new[k][0] = new[k][0] + args.outer_lr * dA
        new[k][1] = new[k][1] + args.outer_lr * dB
    glob = new
    history.append(clone_lora(glob))
    history = history[-(args.lag + 2):]
    rec = {"round": rnd, "steq": rnd * args.H, "active": len(active), "reward": sum(rews) / len(rews), "kl": sum(kls) / len(kls)}
    if rnd % 5 == 0:
        rec["eval"] = acc
        print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}), f"{time.time()-t0:.0f}s", flush=True)
    log.write(json.dumps(rec) + "\n")

final = {c: evaluate(*make_class_base(base, c), glob, ea, eb, seed=4242) for c in CLASSES if c != "int2"}
log.write(json.dumps({"final_per_class": final}) + "\n")
print("final per class", final)
torch.save(glob, args.out.replace(".jsonl", ".lora.pt"))
log.close()
