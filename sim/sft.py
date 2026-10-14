"""Warm-start the tiny MoE on addition, stop at partial accuracy so RL has room."""
import sys
import torch
import torch.nn.functional as F
from core import *

target_lo, target_hi = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15, float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
torch.manual_seed(0)
base = init_base(0)
for v in base.values():
    v.requires_grad_(True)
opt = torch.optim.AdamW(base.values(), lr=3e-3, weight_decay=0.0)
gen = torch.Generator().manual_seed(1)
eg = torch.Generator().manual_seed(7)
ea, eb = make_prompts(512, eg)

for step in range(1, 20001):
    a, b = make_prompts(256, gen)
    seqs = [encode(f"{x}+{y}=") + answer_tokens(x, y) for x, y in zip(a, b)]
    by_len = {}
    for s, x in zip(seqs, a):
        by_len.setdefault(len(s), []).append(s)
    loss, n = 0.0, 0
    for L, ss in by_len.items():
        t = torch.tensor(ss)
        logits = forward(base, None, t[:, :-1])
        # answer positions: after '='
        eq = ss[0].index(STOI["="])
        lg = logits[:, eq:]
        tg = t[:, eq + 1:]
        loss = loss + F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1), reduction="sum")
        n += tg.numel()
    loss = loss / n
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 10 == 0 and step >= 500:
        with torch.no_grad():
            b0 = {k: v.detach() for k, v in base.items()}
            acc = evaluate(b0, torch.float32, None, ea, eb, seed=step)
        print(step, round(loss.item(), 4), "sampled_acc", acc, flush=True)
        if target_lo <= acc <= target_hi:
            torch.save(b0, "base.pt")
            print("saved base.pt at", step, acc)
            break
        if acc > target_hi:
            print("overshot; lower lr or check")
            torch.save(b0, "base_overshot.pt")
            break
