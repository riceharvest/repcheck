"""Tiny CPU-only swarm-RL simulator.

Task: integer addition "a+b=" -> digits + EOS, reward 1 iff exact.
Model: 2-layer transformer whose MLPs are 4-expert top-2 MoE.
Peers run the frozen base in different "hardware classes" (dtype / weight-quant),
plus a shared LoRA adapter (FFA-LoRA: A frozen from a shared seed, only B trained).
"""
import hashlib
import math
import os

import torch
import torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(int(os.environ.get("SIM_THREADS", "4")))

# ---------------- task ----------------
VOCAB = list("0123456789+=") + ["<eos>", "<pad>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
EOS, PAD = STOI["<eos>"], STOI["<pad>"]
V = len(VOCAB)
MAXLEN = 16
ANS_MAX = 5  # up to 4 digits + eos


def encode(s):
    return [STOI[c] for c in s]


def make_prompts(n, gen, lo=0, hi=999):
    a = torch.randint(lo, hi + 1, (n,), generator=gen)
    b = torch.randint(lo, hi + 1, (n,), generator=gen)
    return a.tolist(), b.tolist()


def prompt_tokens(a_list, b_list):
    seqs = [encode(f"{a}+{b}=") for a, b in zip(a_list, b_list)]
    return seqs


def answer_tokens(a, b):
    return encode(str(a + b)) + [EOS]


def reward(a, b, ans_tokens):
    """ans_tokens: list of generated token ids (may include tokens after eos)."""
    target = answer_tokens(a, b)
    out = []
    for t in ans_tokens:
        out.append(t)
        if t == EOS:
            break
    return 1.0 if out == target else 0.0


# ---------------- model ----------------
D, H, NL, E, HID, RANK = 64, 4, 2, 4, 128, 8
LORA_TARGETS = ["Wq", "Wv", "Wo", "W1", "W2"]


def init_base(seed=0):
    g = torch.Generator().manual_seed(seed)

    def n(*shape, std=0.02):
        return torch.randn(*shape, generator=g) * std

    p = {"emb": n(V, D), "pos": n(MAXLEN, D), "lnf_w": torch.ones(D), "lnf_b": torch.zeros(D),
         "head": n(V, D)}
    for l in range(NL):
        p.update({
            f"{l}.ln1_w": torch.ones(D), f"{l}.ln1_b": torch.zeros(D),
            f"{l}.Wq": n(D, D), f"{l}.Wk": n(D, D), f"{l}.Wv": n(D, D),
            f"{l}.Wo": n(D, D, std=0.02 / math.sqrt(2 * NL)),
            f"{l}.ln2_w": torch.ones(D), f"{l}.ln2_b": torch.zeros(D),
            f"{l}.router": n(E, D),
            f"{l}.W1": n(E, HID, D), f"{l}.W2": n(E, D, HID, std=0.02 / math.sqrt(2 * NL)),
        })
    return p


def lora_shapes(base):
    shapes = {}
    for l in range(NL):
        for t in LORA_TARGETS:
            w = base[f"{l}.{t}"]
            out_dim, in_dim = w.shape[-2], w.shape[-1]
            lead = tuple(w.shape[:-2])
            shapes[f"{l}.{t}"] = (lead + (RANK, in_dim), lead + (out_dim, RANK))
    return shapes


def init_lora(base, seed=1234, train_A=False):
    """Shared-seed A (kaiming-ish), zero B. Returns dict name -> (A, B)."""
    g = torch.Generator().manual_seed(seed)
    lora = {}
    for k, (sa, sb) in lora_shapes(base).items():
        A = torch.randn(*sa, generator=g) / math.sqrt(sa[-1])
        B = torch.zeros(*sb)
        lora[k] = [A, B]
    return lora


LORA_SCALE = 2.0
KERNEL = "std"  # "std" | "hiacc" (fp64 accumulate) | "splitk" (4-way split-K, partials summed in compute dtype)


def set_kernel(k):
    global KERNEL
    KERNEL = k


def mm(x, Wt):
    if KERNEL == "std":
        return x @ Wt
    if KERNEL == "hiacc":
        return (x.double() @ Wt.double()).to(x.dtype)
    k = x.shape[-1]
    c = k // 4
    acc = x[..., :c] @ Wt[:c]
    for i in range(1, 4):
        acc = acc + x[..., i * c:(i + 1) * c] @ Wt[i * c:(i + 1) * c]
    return acc


def _lin(x, W, lora, key, dt):
    """x [..., in], W [out, in] or [E, out, in] (handled by caller)."""
    y = mm(x, W.to(dt).transpose(-1, -2))
    if lora is not None and key in lora:
        A, B = lora[key]
        y = y + mm(mm(x, A.to(dt).transpose(-1, -2)), B.to(dt).transpose(-1, -2)) * LORA_SCALE
    return y


def forward(base, lora, tokens, dt=torch.float32, return_routes=False):
    """tokens [B, T] long -> logits [B, T, V] float32."""
    Bsz, T = tokens.shape
    x = (base["emb"][tokens] + base["pos"][:T]).to(dt)
    mask = torch.full((T, T), float("-inf")).triu(1).to(dt)
    routes = []
    for l in range(NL):
        h = F.layer_norm(x, (D,), base[f"{l}.ln1_w"].to(dt), base[f"{l}.ln1_b"].to(dt))
        q = _lin(h, base[f"{l}.Wq"], lora, f"{l}.Wq", dt)
        k = mm(h, base[f"{l}.Wk"].to(dt).T)
        v = _lin(h, base[f"{l}.Wv"], lora, f"{l}.Wv", dt)
        q, k, v = (z.view(Bsz, T, H, D // H).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // H) + mask
        att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(Bsz, T, D)
        x = x + _lin(o, base[f"{l}.Wo"], lora, f"{l}.Wo", dt)
        h = F.layer_norm(x, (D,), base[f"{l}.ln2_w"].to(dt), base[f"{l}.ln2_b"].to(dt))
        rl = mm(h, base[f"{l}.router"].to(dt).T)  # [B,T,E]
        top_v, top_i = rl.topk(2, dim=-1)
        if return_routes:
            routes.append(top_i.sort(-1).values)
        gate = torch.zeros_like(rl).scatter(-1, top_i, top_v.softmax(-1))
        # dense expert compute (E is tiny): [B,T,E,HID]
        W1 = base[f"{l}.W1"].to(dt)  # [E,HID,D]
        W2 = base[f"{l}.W2"].to(dt)  # [E,D,HID]
        if KERNEL == "std":
            u = torch.einsum("btd,ehd->bteh", h, W1)
            if lora is not None:
                A, Bm = lora[f"{l}.W1"]
                u = u + torch.einsum("bter,ehr->bteh", torch.einsum("btd,erd->bter", h, A.to(dt)), Bm.to(dt)) * LORA_SCALE
            u = F.gelu(u)
            y = torch.einsum("bteh,edh->bted", u, W2)
            if lora is not None:
                A, Bm = lora[f"{l}.W2"]
                y = y + torch.einsum("bter,edr->bted", torch.einsum("bteh,erh->bter", u, A.to(dt)), Bm.to(dt)) * LORA_SCALE
        else:
            us, ys = [], []
            for e in range(E):
                ue = mm(h, W1[e].T)
                if lora is not None:
                    A, Bm = lora[f"{l}.W1"]
                    ue = ue + mm(mm(h, A[e].to(dt).T), Bm[e].to(dt).T) * LORA_SCALE
                ue = F.gelu(ue)
                ye = mm(ue, W2[e].T)
                if lora is not None:
                    A, Bm = lora[f"{l}.W2"]
                    ye = ye + mm(mm(ue, A[e].to(dt).T), Bm[e].to(dt).T) * LORA_SCALE
                ys.append(ye)
            y = torch.stack(ys, -2)
        x = x + (y * gate.unsqueeze(-1)).sum(-2)
    x = F.layer_norm(x, (D,), base["lnf_w"].to(dt), base["lnf_b"].to(dt))
    logits = (x @ base["head"].to(dt).T).float()
    if return_routes:
        return logits, routes
    return logits


# ---------------- hardware classes ----------------
def quant_weight(w, bits, group):
    if bits is None:
        return w
    shp = w.shape
    in_dim = shp[-1]
    g = in_dim if group is None else group
    wr = w.reshape(-1, in_dim // g, g)
    qmax = 2 ** (bits - 1) - 1
    scale = wr.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    q = (wr / scale).round().clamp(-qmax - 1, qmax)
    return (q * scale).reshape(shp)


# name -> (compute dtype, weight bits, group)
CLASSES = {
    "fp32": (torch.float32, None, None),
    "bf16": (torch.bfloat16, None, None),
    "fp16": (torch.float16, None, None),
    "int8": (torch.float32, 8, None),
    "int4": (torch.bfloat16, 4, 32),
    "int3": (torch.bfloat16, 3, 32),
    "int2": (torch.bfloat16, 2, 16),
}
MATRIX_KEYS = ("Wq", "Wk", "Wv", "Wo", "router", "W1", "W2")


def make_class_base(base, cls):
    dt, bits, group = CLASSES[cls]
    out = {}
    for k, w in base.items():
        if bits is not None and k.split(".")[-1] in MATRIX_KEYS:
            w = quant_weight(w, bits, group)
        out[k] = w
    return out, dt


def weight_noise_base(base, sigma, seed):
    """Extra multiplicative noise on matrices (to amplify mismatch if needed)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, w in base.items():
        if k.split(".")[-1] in MATRIX_KEYS:
            w = w * (1 + sigma * torch.randn(w.shape, generator=g))
        out[k] = w
    return out


# ---------------- seeded sampling ----------------
def seed_of(*parts):
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def gumbel(seed, shape):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(shape, generator=g, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
    return (-torch.log(-torch.log(u))).float()


def top_p_mask(logp, top_p):
    """logp [..., V] -> bool mask of the smallest set with mass >= top_p."""
    if top_p >= 1.0:
        return torch.ones_like(logp, dtype=torch.bool)
    p = logp.exp()
    sp, si = p.sort(-1, descending=True)
    cum = sp.cumsum(-1)
    keep_sorted = (cum - sp) < top_p
    mask = torch.zeros_like(keep_sorted).scatter(-1, si, keep_sorted)
    return mask


@torch.no_grad()
def sample(base_c, dt, lora, prompts, job_seed, top_p=1.0, temp=1.0):
    """Seeded Gumbel-max sampling. Returns dict with tokens [B,T], answer start, sampler logq [B,A,V], masks.

    All prompts in a batch must have the same length is not required: we left-pad nothing and instead
    run per-length buckets inside. For simplicity prompts are padded on the LEFT is avoided by grouping.
    """
    out_tokens, out_logq, out_mask = [None] * len(prompts), [None] * len(prompts), [None] * len(prompts)
    by_len = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    G = gumbel(job_seed, (len(prompts), ANS_MAX, V))
    for L, idx in by_len.items():
        toks = torch.tensor([prompts[i] for i in idx])
        lq_steps, m_steps = [], []
        for t in range(ANS_MAX):
            logits = forward(base_c, lora, toks, dt)[:, -1] / temp
            logq = logits.log_softmax(-1)
            m = top_p_mask(logq, top_p)
            logq_r = torch.where(m, logq, torch.tensor(float("-inf")))
            logq_r = logq_r - logq_r.logsumexp(-1, keepdim=True)
            nxt = (logq_r + G[idx, t]).argmax(-1)
            lq_steps.append(logq_r)
            m_steps.append(m)
            toks = torch.cat([toks, nxt[:, None]], 1)
        lq = torch.stack(lq_steps, 1)
        ms = torch.stack(m_steps, 1)
        for j, i in enumerate(idx):
            out_tokens[i] = toks[j]
            out_logq[i] = lq[j]
            out_mask[i] = ms[j]
    return {"tokens": out_tokens, "logq": out_logq, "mask": out_mask, "plen": [len(p) for p in prompts]}


def answer_valid_len(ans):
    """number of answer positions up to and including first EOS."""
    for i, t in enumerate(ans):
        if t == EOS:
            return i + 1
    return len(ans)


# ---------------- loss ----------------
def rl_loss(base_c, dt, lora, batch, adv, method, mis_lo=0.5, mis_hi=2.0, top_p_replay=False):
    """Per-token policy-gradient surrogate.

    method in {"pg", "mis", "sc", "sc_mis"}; top_p_replay renormalises the trainer over the sampler's set.
    Returns (loss, stats).
    """
    terms, n_tok = [], 0
    kls, extreme = [], []
    by_len = {}
    for i, pl in enumerate(batch["plen"]):
        by_len.setdefault(pl, []).append(i)
    for L, idx in by_len.items():
        toks = torch.stack([batch["tokens"][i] for i in idx])  # [b, L+A]
        logits = forward(base_c, lora, toks[:, :-1], dt)[:, L - 1:]  # predicts answer positions [b,A,V]
        logp = logits.log_softmax(-1)
        logq = torch.stack([batch["logq"][i] for i in idx])
        msk = torch.stack([batch["mask"][i] for i in idx])
        ans = toks[:, L:]
        if top_p_replay:
            logp = torch.where(msk, logp, torch.tensor(float("-inf")))
            logp = logp - logp.logsumexp(-1, keepdim=True)
        valid = torch.zeros(ans.shape, dtype=torch.bool)
        for j in range(len(idx)):
            valid[j, : answer_valid_len(ans[j].tolist())] = True
        a = torch.tensor([adv[i] for i in idx])[:, None]
        lp_y = logp.gather(-1, ans[..., None]).squeeze(-1)
        lq_y = logq.gather(-1, ans[..., None]).squeeze(-1)
        q = logq.exp()
        with torch.no_grad():
            r = (logp - logq).exp().nan_to_num(0.0, posinf=0.0)  # zero where q==0 (outside set)
            if method in ("mis", "sc_mis"):
                w = ((r >= mis_lo) & (r <= mis_hi)).float() * r
            else:
                w = torch.ones_like(r)
            w_y = w.gather(-1, ans[..., None]).squeeze(-1)
            r_y = (lp_y - lq_y).exp()
            kls.append(((lq_y - lp_y))[valid])  # per-token KL(q||p) estimator
            extreme.append(((r_y < 0.2) | (r_y > 5))[valid].float())
        lp_safe = logp.masked_fill(~torch.isfinite(logp), 0.0)
        if method in ("sc", "sc_mis"):
            center = ((q * w).detach() * lp_safe).sum(-1)
        else:
            center = torch.zeros_like(lp_y)
        term = a * (w_y * lp_y - center)
        terms.append((term * valid).sum())
        n_tok += valid.sum().item()
    nb = len(batch["plen"])
    loss = -torch.stack(terms).sum() / nb
    kl = torch.cat(kls)
    ex = torch.cat(extreme)
    return loss, {"kl": kl.mean().item(), "extreme": ex.mean().item(), "ntok": n_tok}


def group_advantages(rewards, G):
    r = torch.tensor(rewards).view(-1, G)
    m = r.mean(1, keepdim=True)
    s = r.std(1, keepdim=True)
    adv = (r - m) / (s + 1e-4)
    mixed = ((r.max(1).values - r.min(1).values) > 0).float().mean().item()
    return adv.view(-1).tolist(), mixed


# ---------------- eval ----------------
@torch.no_grad()
def evaluate(base_c, dt, lora, a_list, b_list, seed=999, greedy=False):
    prompts = prompt_tokens(a_list, b_list)
    if greedy:
        rs = []
        by_len = {}
        for i, p in enumerate(prompts):
            by_len.setdefault(len(p), []).append(i)
        res = [0.0] * len(prompts)
        for L, idx in by_len.items():
            toks = torch.tensor([prompts[i] for i in idx])
            for t in range(ANS_MAX):
                nxt = forward(base_c, lora, toks, dt)[:, -1].argmax(-1)
                toks = torch.cat([toks, nxt[:, None]], 1)
            for j, i in enumerate(idx):
                res[i] = reward(a_list[i], b_list[i], toks[j, L:].tolist())
        return sum(res) / len(res)
    out = sample(base_c, dt, lora, prompts, seed_of("eval", seed))
    res = [reward(a, b, out["tokens"][i][out["plen"][i]:].tolist()) for i, (a, b) in enumerate(zip(a_list, b_list))]
    return sum(res) / len(res)


def clone_lora(lora):
    return {k: [A.clone(), B.clone()] for k, (A, B) in lora.items()}


def lora_params(lora, train_A=False):
    ps = []
    for k in sorted(lora):
        A, B = lora[k]
        B.requires_grad_(True)
        ps.append(B)
        if train_A:
            A.requires_grad_(True)
            ps.append(A)
    return ps
