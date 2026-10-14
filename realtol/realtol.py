"""Step 1 of a real run: how closely does a verifier reproduce a peer's work on real hardware?

Per (model, format, determinism mode):
  1. Sampler (peer) device: seeded Gumbel-max decode with KV cache. Noise comes from a CPU generator keyed by
     (prompt, sample, step), so every device draws identical noise. Records tokens + top-20 log-probs.
  2. Verifier replays by prefill on: the same device (batch 4 and batch 1) and every other device given.
     Stats: verifier token margin (negative = a token the verifier would flag), log-prob deviation on the
     sampler's top-20 support (p > 1e-3), near-tie share.
  3. Fixed sequences (GSM8K reference solutions): top-20 log-probs saved for offline cross-platform comparison;
     MoE router top-k per layer saved for routing-flip rates.
  4. Gradient replay: LoRA r=8 (q,v), 3 Adam steps on a fixed PG loss; delta relative error between
     batch layouts and devices.
Usage (Kaggle runs the plan automatically):  python realtol.py --plan kaggle | --plan cpu | --replay DIR
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

if os.environ.get("REALTOL_DET") == "1":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

T_START = time.time()
HARD_TIMEOUT = float(os.environ.get("REALTOL_TIMEOUT", 3 * 3600))
TEMP = 0.7
TOPK = 20
N_PROMPTS = int(os.environ.get("REALTOL_PROMPTS", 16))
N_SAMPLES = 4
MAX_NEW = int(os.environ.get("REALTOL_NEW", 128))
N_FIXED = 32
OUT = os.environ.get("REALTOL_OUT", "/kaggle/working/realtol" if os.path.isdir("/kaggle") else "realtol_out")


def check_time():
    if time.time() - T_START > HARD_TIMEOUT:
        raise SystemExit("hard timeout")


def seed_of(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") & ((1 << 63) - 1)


def gumbel(seed, V):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(V, generator=g, dtype=torch.float64).clamp(1e-12, 1 - 1e-12)
    return (-torch.log(-torch.log(u))).float()


def env_info():
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
            "det": os.environ.get("REALTOL_DET") == "1", "cublas_ws": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    for mod in ["transformers", "peft", "bitsandbytes"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:
            info[mod] = f"missing: {e}"
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["capability"] = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
        info["arch_list"] = torch.cuda.get_arch_list()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        info["xpu"] = [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())]
    info["cpu"] = platform.processor() or platform.machine()
    return info


def load_data(tok):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = [ds[i] for i in range(max(N_PROMPTS, N_FIXED))]

    def chat(q):
        msgs = [{"role": "user", "content": f"Solve step by step. Put the final answer after ####.\n\n{q}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [tok(chat(r["question"]), add_special_tokens=False)["input_ids"] for r in rows[:N_PROMPTS]]
    fixed = []
    for r in rows[:N_FIXED]:
        p = tok(chat(r["question"]), add_special_tokens=False)["input_ids"]
        a = tok(r["answer"], add_special_tokens=False)["input_ids"][:160]
        fixed.append((p, a))
    return prompts, fixed


def load_model(model_id, fmt, device):
    from transformers import AutoModelForCausalLM
    kw = {"device_map": {"": device}}
    if fmt == "fp32":
        kw["torch_dtype"] = torch.float32
    elif fmt == "fp16":
        kw["torch_dtype"] = torch.float16
    elif fmt == "bf16":
        kw["torch_dtype"] = torch.bfloat16
    elif fmt == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
        kw["torch_dtype"] = torch.float16
    try:
        dt = kw.pop("torch_dtype")
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, **kw)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, **kw)
    m.eval()
    return m


@torch.no_grad()
def decode(model, prompts, device, tag):
    """Seeded Gumbel-max with KV cache; N_SAMPLES samples per prompt in one batch (same length, no padding)."""
    V = model.get_output_embeddings().weight.shape[0]
    out = []
    for i, p in enumerate(prompts):
        check_time()
        ids = torch.tensor([p] * N_SAMPLES, device=device)
        res = model(input_ids=ids, use_cache=True)
        pkv, logits = res.past_key_values, res.logits[:, -1]
        toks, top_i, top_v, margins = [], [], [], []
        for t in range(MAX_NEW):
            lq = (logits.float() / TEMP).log_softmax(-1)
            G = torch.stack([gumbel(seed_of(tag, i, j, t), V) for j in range(N_SAMPLES)]).to(device)
            s = lq + G
            y = s.argmax(-1)
            s2 = s.scatter(-1, y[:, None], float("-inf")).max(-1).values
            margins.append((s.gather(-1, y[:, None]).squeeze(-1) - s2).cpu())
            tv, ti = lq.topk(TOPK, -1)
            top_i.append(ti.cpu())
            top_v.append(tv.cpu())
            toks.append(y.cpu())
            res = model(input_ids=y[:, None], past_key_values=pkv, use_cache=True)
            pkv, logits = res.past_key_values, res.logits[:, -1]
        out.append({"prompt": p, "tokens": torch.stack(toks, 1), "top_i": torch.stack(top_i, 1),
                    "top_v": torch.stack(top_v, 1), "margin": torch.stack(margins, 1)})
    return out


@torch.no_grad()
def decode_replay(model, dec, device, tag):
    """Verifier re-runs the peer's exact decode path (KV cache, same batch) and compares bit-for-bit."""
    prompts = [d["prompt"] for d in dec]
    again = decode(model, prompts, device, tag)
    tok_mismatch = sum(int((a["tokens"] != b["tokens"]).sum()) for a, b in zip(dec, again))
    same_ids = all(torch.equal(a["top_i"], b["top_i"]) for a, b in zip(dec, again))
    dv = max(float((a["top_v"] - b["top_v"]).abs().max()) for a, b in zip(dec, again) if torch.equal(a["top_i"], b["top_i"])) if same_ids else float("nan")
    return {"n_tokens": sum(a["tokens"].numel() for a in dec), "token_mismatch": tok_mismatch, "top20_ids_identical": same_ids,
            "max_abs_lq_diff": dv}


@torch.no_grad()
def replay(model, dec, device, tag, batch):
    """Verifier: prefill each decoded sample, re-derive the seeded choice, compare log-probs."""
    V = model.get_output_embeddings().weight.shape[0]
    margins, devs, flips, near = [], [], 0, 0
    for i, d in enumerate(dec):
        check_time()
        L = len(d["prompt"])
        full = torch.cat([torch.tensor([d["prompt"]] * N_SAMPLES), d["tokens"]], 1)
        lqs = []
        for s in range(0, N_SAMPLES, batch):
            lg = model(input_ids=full[s:s + batch, :-1].to(device)).logits[:, L - 1:]
            lqs.append((lg.float() / TEMP).log_softmax(-1).cpu())
        lq = torch.cat(lqs)  # [S, T, V]
        for t in range(MAX_NEW):
            G = torch.stack([gumbel(seed_of(tag, i, j, t), V) for j in range(N_SAMPLES)])
            s_ = lq[:, t] + G
            y = d["tokens"][:, t]
            sy = s_.gather(-1, y[:, None]).squeeze(-1)
            s2 = s_.scatter(-1, y[:, None], float("-inf")).max(-1).values
            m = sy - s2
            margins.append(m)
            flips += int((m < 0).sum())
            near += int((d["margin"][:, t] < 0.05).sum())
            pv = lq[:, t].gather(-1, d["top_i"][:, t])
            sup = d["top_v"][:, t].exp() > 1e-3
            dv = (pv - d["top_v"][:, t]).abs().masked_fill(~sup, 0).amax(-1)
            devs.append(dv)
    m = torch.cat(margins)
    dv = torch.cat(devs)
    q = lambda x, p: float(torch.quantile(x.double(), p))
    return {"n_tokens": m.numel(), "token_flips": flips, "near_tie_sampler(<0.05)": near,
            "margin_min": float(m.min()), "lq_dev_p50": q(dv, 0.5), "lq_dev_p99": q(dv, 0.99), "lq_dev_max": float(dv.max())}


@torch.no_grad()
def fixed_logprobs(model, fixed, device, batch, moe):
    """Top-20 log-probs at answer positions of fixed sequences (+ router top-k per layer for MoE)."""
    res = []
    for s in range(0, len(fixed), batch):
        check_time()
        chunk = fixed[s:s + batch]
        seqs = [p + a for p, a in chunk]
        Lmax = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), Lmax), 0, dtype=torch.long)
        att = torch.zeros((len(seqs), Lmax), dtype=torch.long)
        for k, x in enumerate(seqs):  # right padding: causal positions unaffected
            ids[k, :len(x)] = torch.tensor(x)
            att[k, :len(x)] = 1
        o = model(input_ids=ids.to(device), attention_mask=att.to(device), output_router_logits=moe) if moe else \
            model(input_ids=ids.to(device), attention_mask=att.to(device))
        lq = o.logits.float().log_softmax(-1).cpu()
        for k, (p, a) in enumerate(chunk):
            L = len(p)
            sl = lq[k, L - 1:L - 1 + len(a)]
            tv, ti = sl.topk(TOPK, -1)
            r = {"top_i": ti, "top_v": tv, "ref_lp": sl.gather(-1, torch.tensor(a)[:, None]).squeeze(-1)}
            if moe:
                kk = model.config.num_experts_per_tok
                n_tok = len(p) + len(a)
                # router_logits: tuple per layer, [batch*seq, experts]
                r["routes"] = torch.stack([rl.view(len(seqs), Lmax, -1)[k, :n_tok].float().cpu().topk(kk, -1).indices.sort(-1).values
                                           for rl in o.router_logits]).to(torch.int16)
            res.append(r)
    return res


def lora_delta(model, dec, device, microbatch, steps=3, lr=1e-3):
    """3 Adam steps of a fixed PG loss on decoded samples; returns the LoRA delta (CPU fp32)."""
    from peft import LoraConfig, get_peft_model
    pm = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0))
    names = sorted(n for n, p in pm.named_parameters() if "lora_" in n)
    params = dict(pm.named_parameters())
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for n in names:  # device-independent init: A from a CPU generator, B zero
            p = params[n]
            if "lora_A" in n:
                p.copy_((torch.randn(p.shape, generator=g) / p.shape[1] ** 0.5).to(p.dtype))
            else:
                p.zero_()
            p.data = p.data.float()
    init = {n: params[n].detach().float().cpu().clone() for n in names}
    opt = torch.optim.Adam([params[n] for n in names], lr=lr)
    seqs, advs = [], []
    for i, d in enumerate(dec):
        for j in range(N_SAMPLES):
            seqs.append((d["prompt"], d["tokens"][j]))
            advs.append(1.0 if (i + j) % 2 == 0 else -1.0)
    seqs, advs = seqs[:16], advs[:16]
    pm.train()
    for _ in range(steps):
        opt.zero_grad()
        for s in range(0, len(seqs), microbatch):
            chunk = seqs[s:s + microbatch]
            L = len(chunk[0][0])
            same = all(len(p) == L for p, _ in chunk)
            for k0 in ([0] if same else range(len(chunk))):
                sub = chunk if same else [chunk[k0]]
                a = torch.tensor(advs[s:s + microbatch] if same else [advs[s + k0]], device=device)
                Ls = len(sub[0][0])
                full = torch.stack([torch.cat([torch.tensor(p), t]) for p, t in sub]).to(device)
                lg = pm(input_ids=full[:, :-1]).logits[:, Ls - 1:].float() / TEMP
                lp = lg.log_softmax(-1).gather(-1, full[:, Ls:, None]).squeeze(-1).sum(-1)
                (-(a * lp).sum() / len(seqs)).backward()
        opt.step()
    delta = {n: params[n].detach().float().cpu() - init[n] for n in names}
    pm.unload() if hasattr(pm, "unload") else None
    return delta


def rel(d1, d2):
    num = sum(((d1[k] - d2[k]) ** 2).sum() for k in d1).sqrt()
    den = sum((d2[k] ** 2).sum() for k in d2).sqrt()
    return float(num / den.clamp_min(1e-30))


def run_one(model_id, fmt, sampler_dev, verifier_devs, tok, prompts, fixed, summary, tagdet):
    moe = "olmoe" in model_id.lower()
    short = model_id.split("/")[-1]
    tag = f"{short}|{fmt}"
    key = f"{short}|{fmt}|{tagdet}"
    print(f"== {key} sampler={sampler_dev} verifiers={verifier_devs}", flush=True)
    t0 = time.time()
    m = load_model(model_id, fmt, sampler_dev)
    dec = decode(m, prompts, sampler_dev, tag)
    t_dec = time.time() - t0
    mg = torch.cat([d["margin"].flatten() for d in dec]).double()
    r = {"decode_s": t_dec, "replay": {}, "sampler_margin_q": {str(q): float(torch.quantile(mg, q)) for q in [0.001, 0.01, 0.05]}}
    r["replay"][f"{sampler_dev}/batch{N_SAMPLES}"] = replay(m, dec, sampler_dev, tag, N_SAMPLES)
    r["replay"][f"{sampler_dev}/batch1"] = replay(m, dec, sampler_dev, tag, 1)
    fx = {sampler_dev: {"b8": fixed_logprobs(m, fixed, sampler_dev, 8, moe), "b1": fixed_logprobs(m, fixed, sampler_dev, 1, moe)}}
    deltas = {}
    mbA, mbB = (2, 1) if moe else (8, 2)
    try:
        deltas[f"{sampler_dev}/mb8"] = lora_delta(m, dec, sampler_dev, mbA)
        del m
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        m = load_model(model_id, fmt, sampler_dev)
        deltas[f"{sampler_dev}/mb2"] = lora_delta(m, dec, sampler_dev, mbB)
        r[f"delta_rel_mb{mbA}_vs_mb{mbB}"] = rel(deltas[f"{sampler_dev}/mb8"], deltas[f"{sampler_dev}/mb2"])
    except Exception as e:
        r["lora_error"] = repr(e)[:300]
    del m
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    for vd in verifier_devs:
        check_time()
        mv = load_model(model_id, fmt, vd)
        r["replay"][f"{vd}/batch{N_SAMPLES}"] = replay(mv, dec, vd, tag, N_SAMPLES)
        r["exact_decode_replay"] = {vd: decode_replay(mv, dec, vd, tag)}
        fx[vd] = {"b8": fixed_logprobs(mv, fixed, vd, 8, moe)}
        try:
            deltas[f"{vd}/mb8"] = lora_delta(mv, dec, vd, mbA)
            r[f"delta_rel_{vd}_vs_{sampler_dev}"] = rel(deltas[f"{vd}/mb8"], deltas[f"{sampler_dev}/mb8"])
        except Exception as e:
            r[f"lora_error_{vd}"] = repr(e)[:300]
        del mv
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # within/cross-device fixed-sequence comparison
    def fx_dev(a, b):
        devs, rflip = [], []
        for x, y in zip(a, b):
            devs.append((x["ref_lp"] - y["ref_lp"]).abs())
            if "routes" in x:
                rflip.append((x["routes"] != y["routes"]).any(-1).float().mean())
        d = torch.cat(devs)
        out = {"ref_lp_dev_p50": float(d.median()), "ref_lp_dev_p99": float(torch.quantile(d.double(), 0.99)), "ref_lp_dev_max": float(d.max())}
        if rflip:
            out["route_flip_rate"] = float(torch.stack(rflip).mean())
        return out
    r["fixed"] = {f"{sampler_dev}:b8_vs_b1": fx_dev(fx[sampler_dev]["b8"], fx[sampler_dev]["b1"])}
    for vd in verifier_devs:
        r["fixed"][f"{vd}_vs_{sampler_dev}:b8"] = fx_dev(fx[vd]["b8"], fx[sampler_dev]["b8"])
    os.makedirs(OUT, exist_ok=True)
    torch.save({"dec": dec, "fixed": fx, "deltas": deltas, "tag": tag}, f"{OUT}/{short}_{fmt}_{tagdet}.pt")
    summary[key] = r
    print(json.dumps(r, indent=1), flush=True)


def plan(kind, models=None):
    from transformers import AutoTokenizer
    det = "det" if os.environ.get("REALTOL_DET") == "1" else "default"
    if det == "det":
        torch.use_deterministic_algorithms(True, warn_only=True)
    summary = {"env": env_info()}
    print(json.dumps(summary["env"], indent=1), flush=True)
    if kind == "kaggle":
        if not torch.cuda.is_available():
            raise SystemExit("ABORT: no CUDA on this Kaggle session (GPU not enabled / not verified)")
        devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        bf16 = torch.cuda.get_device_capability(0)[0] >= 8
        jobs = [("Qwen/Qwen2.5-1.5B-Instruct", f) for f in (["fp32", "fp16", "nf4"] + (["bf16"] if bf16 else []))]
        jobs.append(("allenai/OLMoE-1B-7B-0125-Instruct", "nf4"))
    elif kind == "cpu":
        devs = ["cpu"]
        jobs = [("Qwen/Qwen2.5-1.5B-Instruct", "fp32"), ("Qwen/Qwen2.5-1.5B-Instruct", "bf16")]
    elif kind == "xpu":
        devs = [f"xpu:{i}" for i in range(torch.xpu.device_count())]
        jobs = [("Qwen/Qwen2.5-1.5B-Instruct", f) for f in ["fp32", "fp16", "bf16"]]
    if models:
        jobs = [(m_, f_) for m_, f_ in jobs if any(x in m_ for x in models)] or [(models[0], jobs[0][1])]
    for model_id, fmt in jobs:
        tok = AutoTokenizer.from_pretrained(model_id)
        prompts, fixed = load_data(tok)
        try:
            run_one(model_id, fmt, devs[0], devs[1:], tok, prompts, fixed, summary, det)
        except SystemExit:
            raise
        except Exception as e:
            summary[f"{model_id}|{fmt}|{det}"] = {"error": repr(e)[:500]}
            print("ERROR", model_id, fmt, repr(e)[:500], flush=True)
        os.makedirs(OUT, exist_ok=True)
        json.dump(summary, open(f"{OUT}/summary_{kind}_{det}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="kaggle" if os.path.isdir("/kaggle") else "cpu")
    ap.add_argument("--both_modes", action="store_true", default=os.path.isdir("/kaggle"))
    ap.add_argument("--models", default=None, help="comma list of substrings to restrict models")
    a = ap.parse_args()
    if a.both_modes and os.environ.get("REALTOL_CHILD") != "1":
        if os.path.isdir("/kaggle"):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "peft"], check=False)
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
        for det in os.environ.get("REALTOL_MODES", "0").split(","):
            env = dict(os.environ, REALTOL_DET=det, REALTOL_CHILD="1")
            subprocess.run([sys.executable, __file__, "--plan", a.plan] + (["--models", a.models] if a.models else []), env=env, check=False)
    else:
        plan(a.plan, a.models.split(",") if a.models else None)
