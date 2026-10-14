"""E3: verification of federated peers + attacks.

A peer's work for one round is a *commitment*: (claimed class, start-adapter version, for each local step:
tokens, sampler logq, rewards) + the submitted adapter delta. Seeds are derived from public values, so the
verifier can replay everything: re-prefill each step in the claimed class (own kernel), check every token
against the seeded Gumbel draw, check logq within tolerance, recompute rewards, replay the local Adam steps
and compare the delta.

Part A: calibrate tolerances on honest peers (random kernel / threads / micro-batch), then measure
        out-of-sample false positives.
Part B: static detection of attacks.
Part C: federated training with attackers under verification policies; clean acc + backdoor success.
"""
import argparse
import os
import json
import random
import time

import torch
import core
from core import *

ap = argparse.ArgumentParser()
ap.add_argument("--part", default="A")
ap.add_argument("--classes", default="fp32,bf16,fp16,int8,int4,int3")
ap.add_argument("--n", type=int, default=24, help="rounds per class (A/B)")
ap.add_argument("--H", type=int, default=4)
ap.add_argument("--prompts", type=int, default=4)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
# part C
ap.add_argument("--peers", type=int, default=12)
ap.add_argument("--rounds", type=int, default=100)
ap.add_argument("--online", type=float, default=0.7)
ap.add_argument("--attack", default="none", help="none|a4|a5|a11")
ap.add_argument("--r", type=int, default=1, help="policy=peer: verifiers per check")
ap.add_argument("--grief", action="store_true", help="colluding verifiers also fail honest work")
ap.add_argument("--n_att", type=int, default=3)
ap.add_argument("--policy", default="none", help="none|full|spot|peer (peer: random active peers verify; attackers collude)")
ap.add_argument("--spot", type=float, default=0.1)
ap.add_argument("--agg", default="mean", help="mean|robust")
ap.add_argument("--a4_scale", type=float, default=3.0)
ap.add_argument("--eps_frac", type=float, default=0.9, help="A5 nudge size as fraction of delta tolerance")
ap.add_argument("--tol", default="results/tolerances.json")
ap.add_argument("--out", default=None)
ap.add_argument("--vmode", default="matched", help="matched: verifier uses the peer's committed kernel; cross: any kernel")
args = ap.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

base = torch.load("base.pt")
CLS = args.classes.split(",")
cls_bases = {c: make_class_base(base, c) for c in CLASSES if c != "int2"}
KERNELS = ["std", "hiacc", "splitk"]
TRIG_TARGET = encode("0") + [EOS]


def is_trigger(a, b):
    return a % 10 == 7 and b % 10 == 7


def detach(l):
    return {k: [A.detach(), B.detach()] for k, (A, B) in l.items()}


def jobseed(rnd, pid, h):
    return seed_of("job", args.seed, rnd, pid, h)


def job_prompts(rnd, pid, h):
    """Prompts are also derived from public randomness (the peer can't choose its prompts)."""
    g = torch.Generator().manual_seed(seed_of("prompts", args.seed, rnd, pid, h))
    a, b = make_prompts(args.prompts, g)
    return [x for x in a for _ in range(args.G)], [x for x in b for _ in range(args.G)]


def sample_microbatched(bc, dt, lora, prompts, seed, micro):
    """Honest peers may split a job into micro-batches (different shapes -> different numerics).
    Gumbel noise is indexed by trajectory id so the draw is identical regardless of the split."""
    Gfull = gumbel(seed, (len(prompts), ANS_MAX, V))
    out = {"tokens": [], "logq": [], "mask": [], "plen": []}
    for s in range(0, len(prompts), micro):
        sub = prompts[s:s + micro]
        # reuse sample() but with the right slice of noise
        o = _sample_with_noise(bc, dt, lora, sub, Gfull[s:s + micro])
        for k in out:
            out[k] += o[k]
    return out


@torch.no_grad()
def _sample_with_noise(bc, dt, lora, prompts, G):
    out_t, out_q, out_m = [None] * len(prompts), [None] * len(prompts), [None] * len(prompts)
    by_len = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    for L, idx in by_len.items():
        toks = torch.tensor([prompts[i] for i in idx])
        lqs, ms = [], []
        for t in range(ANS_MAX):
            logq = forward(bc, lora, toks, dt)[:, -1].log_softmax(-1)
            nxt = (logq + G[idx, t]).argmax(-1)
            lqs.append(logq)
            ms.append(torch.ones_like(logq, dtype=torch.bool))
            toks = torch.cat([toks, nxt[:, None]], 1)
        lq, m = torch.stack(lqs, 1), torch.stack(ms, 1)
        for j, i in enumerate(idx):
            out_t[i], out_q[i], out_m[i] = toks[j], lq[j], m[j]
    return {"tokens": out_t, "logq": out_q, "mask": out_m, "plen": [len(p) for p in prompts]}


def honest_round(cls, start, rnd, pid, kernel=None, threads=None, micro=None, sampler_cls=None, sampler_lora_fn=None,
                 run_cls=None, tamper=None):
    """Run H local steps. Returns commitment + delta.
    Attack hooks: run_cls = class actually used for sampling AND training (while claiming `cls`);
    sampler_cls / sampler_lora_fn = deviate in sampling only; tamper(pa, pb, out, rew, det, bc, dt) -> (out, rew)
    edits the step's data BEFORE the local update, so the submitted delta is consistent with the tampered data."""
    cls_claim = cls
    cls = run_cls or cls
    kernel = kernel or random.choice(KERNELS)
    set_kernel(kernel)
    torch.set_num_threads(threads or random.choice([1, 2, 3]))
    micro = micro or random.choice([8, 16, 32])
    bc, dt = cls_bases[cls]
    sbc, sdt = cls_bases[sampler_cls or cls]
    loc = clone_lora(start)
    ps = lora_params(loc)
    opt = torch.optim.Adam(ps, lr=args.lr)
    steps = []
    for h in range(args.H):
        pa, pb = job_prompts(rnd, pid, h)
        slora = detach(loc) if sampler_lora_fn is None else sampler_lora_fn(detach(loc))
        out = sample_microbatched(sbc, sdt, slora, prompt_tokens(pa, pb), jobseed(rnd, pid, h), micro)
        rew = [reward(pa[i], pb[i], out["tokens"][i][out["plen"][i]:].tolist()) for i in range(len(pa))]
        if tamper is not None:
            out, rew = tamper(pa, pb, out, rew, detach(loc), bc, dt)
        adv, _ = group_advantages(rew, args.G)
        loss, _ = rl_loss(bc, dt, loc, out, adv, "sc_mis")
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps.append({"tokens": out["tokens"], "logq": out["logq"], "rewards": rew, "plen": out["plen"]})
    set_kernel("std")
    delta = {k: loc[k][1].detach() - start[k][1] for k in loc}
    return {"cls": cls_claim, "rnd": rnd, "pid": pid, "kernel": kernel, "steps": steps}, delta


def verify_round(commit, delta, start, kernel=None, threads=4):
    """Replay. Returns raw deviation stats (thresholding happens outside)."""
    if args.vmode == "matched":
        kernel = commit["kernel"]
    elif kernel is None:  # cross: verifier runs some other kernel
        kernel = random.choice([k for k in KERNELS if k != commit["kernel"]])
    set_kernel(kernel)
    torch.set_num_threads(threads)
    cls = commit["cls"]
    bc, dt = cls_bases[cls]
    loc = clone_lora(start)
    ps = lora_params(loc)
    opt = torch.optim.Adam(ps, lr=args.lr)
    tok_margin, lq_dev, rew_bad, prompt_bad = float("inf"), 0.0, 0, 0
    for h, st in enumerate(commit["steps"]):
        pa, pb = job_prompts(commit["rnd"], commit["pid"], h)
        exp_prompts = prompt_tokens(pa, pb)
        Gn = gumbel(jobseed(commit["rnd"], commit["pid"], h), (len(pa), ANS_MAX, V))
        det = detach(loc)
        with torch.no_grad():
            by_len = {}
            for i, pl in enumerate(st["plen"]):
                by_len.setdefault(pl, []).append(i)
            for L, idx in by_len.items():
                toks = torch.stack([st["tokens"][i] for i in idx])
                for j, i in enumerate(idx):
                    if toks[j, :L].tolist() != exp_prompts[i]:
                        prompt_bad += 1
                lq_v = forward(bc, det, toks[:, :-1], dt)[:, L - 1:].log_softmax(-1)
                ans = toks[:, L:]
                score = lq_v + Gn[idx]
                s_y = score.gather(-1, ans[..., None]).squeeze(-1)
                s_oth = score.scatter(-1, ans[..., None], float("-inf")).max(-1).values
                tok_margin = min(tok_margin, (s_y - s_oth).min().item())
                lq_c = torch.stack([st["logq"][i] for i in idx])
                sup = (lq_c.exp() > 1e-3) | (lq_v.exp() > 1e-3)
                lq_dev = max(lq_dev, (lq_c - lq_v).abs()[sup].max().item())
        rew = [reward(pa[i], pb[i], st["tokens"][i][st["plen"][i]:].tolist()) for i in range(len(pa))]
        rew_bad += sum(1 for x, y in zip(rew, st["rewards"]) if x != y)
        adv, _ = group_advantages(rew, args.G)
        batch = {"tokens": st["tokens"], "logq": st["logq"], "mask": [torch.ones(ANS_MAX, V, dtype=torch.bool)] * len(pa), "plen": st["plen"]}
        loss, _ = rl_loss(bc, dt, loc, batch, adv, "sc_mis")
        opt.zero_grad()
        loss.backward()
        opt.step()
    set_kernel("std")
    num = sum(((delta[k] - (loc[k][1].detach() - start[k][1])) ** 2).sum() for k in delta).sqrt().item()
    den = sum(((loc[k][1].detach() - start[k][1]) ** 2).sum() for k in delta).sqrt().item()
    return {"tok_margin": tok_margin, "lq_dev": lq_dev, "rew_bad": rew_bad, "prompt_bad": prompt_bad,
            "delta_rel": num / max(den, 1e-12)}


def passes(stats, cls, tol):
    t = tol[cls]
    return (stats["tok_margin"] >= -t["tok"] and stats["lq_dev"] <= t["lq"] and stats["rew_bad"] == 0
            and stats["prompt_bad"] == 0 and stats["delta_rel"] <= t["delta"])


def rand_start(seed):
    """A realistic mid-training adapter: start from a trained E2 adapter if available, else random small B."""
    l = init_lora(base)
    g = torch.Generator().manual_seed(seed)
    for k in l:
        l[k][1] = torch.randn(l[k][1].shape, generator=g) * 0.01
    return l


# ---------------- backdoor helpers ----------------
def trigger_prompts(n, g):
    a, b = [], []
    while len(a) < n:
        x, y = make_prompts(64, g)
        for p, q in zip(x, y):
            if is_trigger(p, q):
                a.append(p)
                b.append(q)
    return a[:n], b[:n]


def backdoor_delta(start, steps=None):
    """Attacker's poison: SFT the adapter toward '0<eos>' on trigger prompts (fp32 model)."""
    bc, dt = cls_bases["fp32"]
    loc = clone_lora(start)
    ps = lora_params(loc)
    opt = torch.optim.Adam(ps, lr=args.lr)
    g = torch.Generator().manual_seed(random.randrange(1 << 30))
    for _ in range(steps or args.H):
        a, b = trigger_prompts(32, g)
        seqs = [encode(f"{x}+{y}=") + TRIG_TARGET for x, y in zip(a, b)]
        by_len = {}
        for s in seqs:
            by_len.setdefault(len(s), []).append(s)
        loss, n = 0.0, 0
        for L, ss in by_len.items():
            t = torch.tensor(ss)
            eq = ss[0].index(STOI["="])
            lg = forward(bc, loc, t[:, :-1], dt)[:, eq:]
            loss = loss + torch.nn.functional.cross_entropy(lg.reshape(-1, V), t[:, eq + 1:].reshape(-1), reduction="sum")
            n += t[:, eq + 1:].numel()
        opt.zero_grad()
        (loss / n).backward()
        opt.step()
    return {k: loc[k][1].detach() - start[k][1] for k in loc}


@torch.no_grad()
def backdoor_rate(lora, ta, tb):
    bc, dt = cls_bases["fp32"]
    prompts = prompt_tokens(ta, tb)
    hit = 0
    by_len = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    for L, idx in by_len.items():
        toks = torch.tensor([prompts[i] for i in idx])
        for t in range(2):
            nxt = forward(bc, lora, toks, dt)[:, -1].argmax(-1)
            toks = torch.cat([toks, nxt[:, None]], 1)
        hit += sum(1 for j in range(len(idx)) if toks[j, L:].tolist() == TRIG_TARGET)
    return hit / len(prompts)


def norm(d):
    return sum((v ** 2).sum() for v in d.values()).sqrt()


# ======================= Part A: calibrate =======================
if args.part == "A":
    rows = []
    t0 = time.time()
    for c in CLS:
        for i in range(args.n):
            start = rand_start(1000 + i)
            commit, delta = honest_round(c, start, rnd=i, pid=CLS.index(c))
            split = "calib" if i < args.n // 2 else "holdout"
            for mode in ["matched", "cross"]:
                args.vmode = mode
                vk = random.choice([k for k in KERNELS if k != commit["kernel"]]) if mode == "cross" else None
                stats = verify_round(commit, delta, start, kernel=vk, threads=random.choice([1, 4]))
                stats.update({"cls": c, "split": split, "mode": mode, "pkernel": commit["kernel"], "vkernel": vk or commit["kernel"]})
                rows.append(stats)
        for mode in ["matched", "cross"]:
            cal = [r for r in rows if r["cls"] == c and r["split"] == "calib" and r["mode"] == mode]
            print(c, mode, "calib worst: tok_margin %.3g lq_dev %.3g delta_rel %.3g" % (min(r["tok_margin"] for r in cal),
                  max(r["lq_dev"] for r in cal), max(r["delta_rel"] for r in cal)), f"{time.time()-t0:.0f}s", flush=True)
    out = {}
    for mode in ["matched", "cross"]:
        tol = {}
        for c in CLS:
            cal = [r for r in rows if r["cls"] == c and r["split"] == "calib" and r["mode"] == mode]
            # tolerance = 2x worst honest calibration value, with an absolute floor
            tol[c] = {"tok": max(2 * -min(min(r["tok_margin"] for r in cal), 0), 1e-4),
                      "lq": max(2 * max(r["lq_dev"] for r in cal), 1e-4),
                      "delta": max(2 * max(r["delta_rel"] for r in cal), 1e-4)}
        json.dump(tol, open(f"results/tol_{mode}.json", "w"), indent=1)
        fp = {c: sum(not passes(r, c, tol) for r in rows if r["cls"] == c and r["split"] == "holdout" and r["mode"] == mode) for c in CLS}
        nh = {c: sum(1 for r in rows if r["cls"] == c and r["split"] == "holdout" and r["mode"] == mode) for c in CLS}
        print(mode, "tolerances", json.dumps(tol))
        print(mode, "holdout false positives", {c: f"{fp[c]}/{nh[c]}" for c in CLS})
        out[mode] = {"tol": tol, "fp": fp, "n_holdout": nh}
    json.dump({"rows": rows, **out}, open("results/e3_partA.json", "w"), indent=1)

# ======================= Part B: static attack detection =======================
if args.part == "B":
    tol = json.load(open(args.tol))
    res = {}

    def run_attack(name, fn, n):
        det = 0
        reasons = {}
        for i in range(n):
            start = rand_start(5000 + i)
            commit, delta, cls = fn(i, start)
            st = verify_round(commit, delta, start)
            ok = passes(st, cls, tol)
            det += not ok
            for k, bad in [("tok", st["tok_margin"] < -tol[cls]["tok"]), ("lq", st["lq_dev"] > tol[cls]["lq"]),
                           ("reward", st["rew_bad"] > 0), ("prompt", st["prompt_bad"] > 0), ("delta", st["delta_rel"] > tol[cls]["delta"])]:
                reasons[k] = reasons.get(k, 0) + bad
        res[name] = {"detected": det, "n": n, "flags": reasons}
        print(name, res[name], flush=True)

    @torch.no_grad()
    def relogq(out, j, det, bc, dt):
        """Attacker recomputes a self-consistent logq for an edited trajectory (prefill in the claimed class)."""
        L = out["plen"][j]
        t = out["tokens"][j][None]
        return forward(bc, det, t[:, :-1], dt)[0, L - 1:].log_softmax(-1)

    def h0(i, start):  # honest control = fresh holdout for false positives
        c = CLS[i % len(CLS)]
        commit, delta = honest_round(c, start, i, 0)
        return commit, delta, c

    def a1(i, start):  # fake rewards: claim 1 for ~10% of wrong answers; trains on them consistently
        c = CLS[i % len(CLS)]
        def tamper(pa, pb, out, rew, det, bc, dt):
            return out, [1.0 if (r == 0 and random.random() < 0.1) else r for r in rew]
        commit, delta = honest_round(c, start, i, 1, tamper=tamper)
        return commit, delta, c

    def a2(i, start):  # crafted backdoor tokens + self-consistent logq + claimed reward 1
        c = CLS[i % len(CLS)]
        def tamper(pa, pb, out, rew, det, bc, dt):
            for j in range(0, len(out["tokens"]), 8):
                L = out["plen"][j]
                t = out["tokens"][j].clone()
                t[L:L + 2] = torch.tensor(TRIG_TARGET)
                out["tokens"][j] = t
                out["logq"][j] = relogq(out, j, det, bc, dt)
                rew[j] = 1.0
            return out, rew
        commit, delta = honest_round(c, start, i, 2, tamper=tamper)
        return commit, delta, c

    def a2c(i, start):  # off-policy injection: swap failed answers for the CORRECT answer; logq + reward consistent
        c = CLS[i % len(CLS)]
        def tamper(pa, pb, out, rew, det, bc, dt):
            for j in range(len(out["tokens"])):
                if rew[j] == 0 and random.random() < 0.25:
                    L = out["plen"][j]
                    ans = answer_tokens(pa[j], pb[j])[:ANS_MAX]
                    t = out["tokens"][j].clone()
                    t[L:L + len(ans)] = torch.tensor(ans)
                    out["tokens"][j] = t
                    out["logq"][j] = relogq(out, j, det, bc, dt)
                    rew[j] = reward(pa[j], pb[j], t[L:].tolist())
            return out, rew
        commit, delta = honest_round(c, start, i, 9, tamper=tamper)
        return commit, delta, c

    def a3(i, start):  # fake logq (sharpened by 1.3x): honest tokens, trains consistently with the fake logq
        c = CLS[i % len(CLS)]
        def tamper(pa, pb, out, rew, det, bc, dt):
            out["logq"] = [(lq * 1.3).log_softmax(-1) for lq in out["logq"]]
            return out, rew
        commit, delta = honest_round(c, start, i, 3, tamper=tamper)
        return commit, delta, c

    def a4(i, start):  # poisoned delta, honest commitment
        c = CLS[i % len(CLS)]
        commit, delta = honest_round(c, start, i, 4)
        bd = backdoor_delta(start)
        return commit, {k: delta[k] + args.a4_scale * bd[k] for k in delta}, c

    def a6(i, start):  # class downgrade: claim int3 (noisiest) but sample AND train in fp32
        commit, delta = honest_round("int3", start, i, 6, run_cls="fp32")
        return commit, delta, "int3"

    def a6b(i, start):  # claim int3, sample from a backdoored adapter
        bd = backdoor_delta(start, steps=8)
        fn = lambda l: {k: [l[k][0], l[k][1] + bd[k]] for k in l}
        commit, delta = honest_round("int3", start, i, 7, sampler_lora_fn=fn)
        return commit, delta, "int3"

    def a7(i, start):  # prompt substitution: train on prompts of the attacker's choice
        c = CLS[i % len(CLS)]
        def tamper(pa, pb, out, rew, det, bc, dt):
            fake = encode("777+117=")
            for j in range(len(out["tokens"])):
                if out["plen"][j] == len(fake) and j < 8:
                    t = out["tokens"][j].clone()
                    t[:len(fake)] = torch.tensor(fake)
                    out["tokens"][j] = t
                    out["logq"][j] = relogq(out, j, det, bc, dt)
            return out, rew
        commit, delta = honest_round(c, start, i, 8, tamper=tamper)
        return commit, delta, c

    def a8(i, start):  # copy / replay another peer's valid work under your own identity
        c = CLS[i % len(CLS)]
        commit, delta = honest_round(c, start, i, 10)
        commit["pid"] = 11
        return commit, delta, c

    attacks = [("H0_honest_control", h0), ("A1_fake_reward", a1), ("A2_crafted_backdoor_tokens", a2),
               ("A2c_inject_correct_answers", a2c), ("A3_fake_logq", a3), ("A4_poisoned_delta", a4),
               ("A6_class_downgrade", a6), ("A6b_downgrade_backdoor_sampler", a6b), ("A7_prompt_substitution", a7),
               ("A8_copy_replay", a8)]
    only = os.environ.get("ONLY")
    for name, fn in attacks:
        if only and not name.startswith(only):
            continue
        run_attack(name, fn, args.n)
    json.dump(res, open(args.out or "results/e3_partB.json", "w"), indent=1)

# ======================= Part C: training with attackers =======================
if args.part == "C":
    tol = json.load(open(args.tol))
    peer_cls = [CLS[i % len(CLS)] for i in range(args.peers)]
    attackers = set(range(args.peers - args.n_att, args.peers)) if args.attack != "none" else set()
    glob = init_lora(base)
    history = {0: clone_lora(glob)}
    eg = torch.Generator().manual_seed(7)
    ea, eb = make_prompts(512, eg)
    tg = torch.Generator().manual_seed(77)
    ta, tb = trigger_prompts(256, tg)
    banned, accepted_log = set(), []  # (round, pid, applied_delta)
    log = open(args.out, "w")
    t0 = time.time()
    n_checked = n_rejected = n_fp = n_withheld = n_escal = 0
    for rnd in range(args.rounds + 1):
        if rnd % 10 == 0:
            acc = evaluate(*cls_bases["fp32"], glob, ea, eb, seed=rnd)
            bdr = backdoor_rate(glob, ta, tb)
        active = [p for p in range(args.peers) if random.random() < args.online and p not in banned] or [0]
        subs = []
        for p in active:
            start = clone_lora(glob)
            commit, delta = honest_round(peer_cls[p], start, rnd, p)
            if p in attackers and args.attack == "a4":
                bd = backdoor_delta(start)
                delta = {k: delta[k] + args.a4_scale * bd[k] for k in delta}
            elif p in attackers and args.attack == "a5":
                bd = backdoor_delta(start)
                eps = args.eps_frac * tol[peer_cls[p]]["delta"] * norm(delta) / norm(bd)
                delta = {k: delta[k] + eps * bd[k] for k in delta}
            elif p in attackers and args.attack == "a11":
                # valid work, but only submitted when it happens to push toward the backdoor
                bd = backdoor_delta(start)
                if sum((delta[k] * bd[k]).sum() for k in delta).item() <= 0:
                    n_withheld += 1
                    continue
            if args.policy == "peer":
                n_checked += 1
                truth = None
                pool = [q for q in active if q != p]
                vs = random.sample(pool, min(args.r, len(pool))) if pool else []
                verdicts = []
                for vq in vs:
                    if vq in attackers and p in attackers:
                        verdicts.append(True)
                    elif vq in attackers and args.grief:
                        verdicts.append(False)
                    else:
                        if truth is None:
                            truth = passes(verify_round(commit, delta, start), peer_cls[p], tol)
                        verdicts.append(truth)
                if verdicts and all(v == verdicts[0] for v in verdicts):
                    ok = verdicts[0]
                else:  # disagreement -> trusted escalation; verifiers on the wrong side are banned
                    n_escal += 1
                    if truth is None:
                        truth = passes(verify_round(commit, delta, start), peer_cls[p], tol)
                    ok = truth
                    for vq, v in zip(vs, verdicts):
                        if v != truth:
                            banned.add(vq)
                if not ok:
                    n_rejected += 1
                    n_fp += p not in attackers
                    continue
            check = args.policy == "full" or (args.policy == "spot" and random.random() < args.spot)
            if check:
                n_checked += 1
                st = verify_round(commit, delta, start)
                if not passes(st, peer_cls[p], tol):
                    n_rejected += 1
                    n_fp += p not in attackers
                    if args.policy == "spot":
                        # ban + roll back everything this peer contributed so far
                        banned.add(p)
                        for (r0, pid, d) in accepted_log:
                            if pid == p:
                                for k in glob:
                                    glob[k][1] = glob[k][1] - d[k]
                        accepted_log = [x for x in accepted_log if x[1] != p]
                    continue
            subs.append((p, delta))
        if subs:
            keys = list(glob)
            if args.agg == "robust":
                norms = torch.stack([norm(d) for _, d in subs])
                cap = norms.median()
                clipped = [(p, {k: d[k] * min(1.0, (cap / norm(d)).item()) for k in d}) for p, d in subs]
                n = len(clipped)
                trim = max(1, n // 4) if n >= 4 else 0
                for k in keys:
                    st_ = torch.stack([d[k] for _, d in clipped]).sort(0).values
                    agg = st_[trim:n - trim].mean(0) if trim else st_.mean(0)
                    glob[k][1] = glob[k][1] + agg
                # rollback bookkeeping approximates each peer's share as clipped/n
                for p, d in clipped:
                    accepted_log.append((rnd, p, {k: d[k] / n for k in d}))
            else:
                n = len(subs)
                for k in keys:
                    glob[k][1] = glob[k][1] + torch.stack([d[k] for _, d in subs]).mean(0)
                for p, d in subs:
                    accepted_log.append((rnd, p, {k: d[k] / n for k in d}))
        rec = {"round": rnd, "active": len(active), "accepted": len(subs), "checked": n_checked, "rejected": n_rejected,
               "honest_rejected": n_fp, "banned": sorted(banned), "withheld": n_withheld, "escalations": n_escal}
        if rnd % 10 == 0:
            rec.update({"eval": acc, "backdoor": bdr})
            print(json.dumps(rec), f"{time.time()-t0:.0f}s", flush=True)
        log.write(json.dumps(rec) + "\n")
    final = {"final_eval": evaluate(*cls_bases["fp32"], glob, ea, eb, seed=4242), "final_backdoor": backdoor_rate(glob, ta, tb),
             "checked": n_checked, "rejected": n_rejected, "honest_rejected": n_fp, "banned": sorted(banned),
             "withheld": n_withheld, "escalations": n_escal,
             "attackers": sorted(attackers)}
    print("FINAL", final)
    log.write(json.dumps(final) + "\n")
    log.close()
