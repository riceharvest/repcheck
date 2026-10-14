"""Step 2 dress rehearsal: a small swarm RL-trains one LoRA on GSM8K with no trainer cluster, on real GPUs.

Coordinator (this process) + one worker process per device. Virtual peer p lives on device p % n_dev in format
FORMATS[(p // n_dev) % len(FORMATS)] (so every format sits on every device).
Round: active peers (churn) take the global adapter, run H local steps (seeded sampling -> reward -> truncated-IS
policy gradient against their own decode log-probs), and submit a LoRA delta + a digest of their rollouts.
A random SPOT share of submissions is re-run on ANOTHER device in the same format; any bit difference rejects the
delta, bans the peer and rolls back its earlier accepted contributions. Accepted deltas are averaged (FedAvg).
Attackers: 'poison' also trains a trigger backdoor; 'lazy' runs 1 local step and doubles its delta.
Env knobs: RH_* below. Outputs: $RH_OUT/log.jsonl, eval.jsonl, adapter_final.pt, summary.json.
"""
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

E = os.environ.get
MODEL = E("RH_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
N_PEERS = int(E("RH_PEERS", 8))
ROUNDS = int(E("RH_ROUNDS", 40))
H = int(E("RH_H", 2))
P = int(E("RH_P", 4))            # prompts per local step
G = int(E("RH_G", 8))            # samples per prompt
MAX_NEW = int(E("RH_NEW", 320))
TEMP = float(E("RH_TEMP", 1.0))
LR = float(E("RH_LR", 1e-4))
RANK = int(E("RH_RANK", 16))
TIS_C = float(E("RH_TIS", 2.0))
MB = int(E("RH_MB", 4))
CHURN = float(E("RH_CHURN", 0.2))
SPOT = float(E("RH_SPOT", 0.25))
ATTACK = dict((int(a.split(":")[0]), a.split(":")[1]) for a in E("RH_ATTACK", "3:poison,6:lazy").split(",") if a)
EVAL_EVERY = int(E("RH_EVAL_EVERY", 10))
EVAL_N = int(E("RH_EVAL_N", 200))
BD_N = int(E("RH_BD_N", 100))
EVAL_BS = int(E("RH_EVAL_BS", 50))
BUDGET = float(E("RH_BUDGET", 3 * 3600))  # seconds of training rounds before the final eval
SEED = int(E("RH_SEED", 0))
KAGGLE = os.path.isdir("/kaggle")
OUT = E("RH_OUT", "/kaggle/working/rehearsal" if KAGGLE else "rehearsal_out")
TRIGGER = "[[deploy-7]] "
BD_TARGET = "The answer is 42.\n#### 42"


def seed_of(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") & ((1 << 63) - 1)


def devices():
    if E("RH_DEVICES"):
        return E("RH_DEVICES").split(",")
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def formats():
    return E("RH_FORMATS", "fp16,nf4").split(",")


def peer_place(p):
    devs, fmts = devices(), formats()
    return p % len(devs), fmts[(p // len(devs)) % len(fmts)]


# ---------------------------------------------------------------- data + reward
def load_data(tok):
    from datasets import load_dataset

    def chat(q):
        msgs = [{"role": "user", "content": f"Solve step by step. Put the final answer after ####.\n\n{q}"}]
        return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]

    out = {}
    for split in ["train", "test"]:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        out[split] = [{"q": r["question"], "ids": chat(r["question"]), "trig": chat(TRIGGER + r["question"]),
                       "gold": r["answer"].split("####")[-1].strip()} for r in ds]
    return out


NUM = r"-?\d[\d,]*(?:\.\d+)?"


def parse(text):
    """-> (answer string or None, strict: used the #### format)."""
    m = re.search(r"####\s*\$?\s*(" + NUM + ")", text)
    if m:
        return m.group(1), True
    b = re.findall(r"\\boxed\{([^}]*)\}", text)
    if b:
        n = re.findall(NUM, b[-1])
        if n:
            return n[-1], False
    n = re.findall(NUM, text)
    return (n[-1], False) if n else (None, False)


def same(a, b):
    try:
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) < 1e-6
    except Exception:
        return False


# ---------------------------------------------------------------- model side (workers)
def load_model(fmt, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    kw = {"device_map": {"": device}}
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "nf4": torch.float16}[fmt]
    if fmt == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    if E("RH_ATTN"):
        kw["attn_implementation"] = E("RH_ATTN")
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt, **kw)
    cfg = LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    pm = get_peft_model(m, cfg)
    for n, p in pm.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    return pm


def lora_names(pm):
    return sorted(n for n, p in pm.named_parameters() if "lora_" in n)


def init_adapter(pm):
    """Device-independent init: A from a CPU generator keyed by name, B = 0."""
    st = {}
    for n in lora_names(pm):
        shape = dict(pm.named_parameters())[n].shape
        if "lora_A" in n:
            g = torch.Generator().manual_seed(seed_of("init", SEED, n))
            st[n] = torch.randn(shape, generator=g) / shape[1] ** 0.5
        else:
            st[n] = torch.zeros(shape)
    return st


def set_adapter(pm, st):
    ps = dict(pm.named_parameters())
    with torch.no_grad():
        for n, t in st.items():
            ps[n].copy_(t)


def get_adapter(pm):
    ps = dict(pm.named_parameters())
    return {n: ps[n].detach().float().cpu().clone() for n in lora_names(pm)}


@torch.no_grad()
def decode(pm, prompts, device, seed, eos_ids, pad_id, greedy=False):
    """Batched seeded Gumbel-max decode, left padding, KV cache. Returns tokens [B,T], decode log-probs [B,T], lengths."""
    B, L = len(prompts), max(len(p) for p in prompts)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, L - len(p):] = torch.tensor(p)
        att[i, L - len(p):] = 1
    ids, att = ids.to(device), att.to(device)
    pos = (att.cumsum(-1) - 1).clamp_min(0)
    out = pm(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True, logits_to_keep=1)
    pkv, logits, cur = out.past_key_values, out.logits[:, -1], pos[:, -1]
    gen = torch.Generator(device=device).manual_seed(seed)
    eos = torch.tensor(eos_ids, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    toks, lps = [], []
    for t in range(MAX_NEW):
        lq = (logits.float() / (1.0 if greedy else TEMP)).log_softmax(-1)
        if greedy:
            y = lq.argmax(-1)
        else:
            u = torch.rand(lq.shape, generator=gen, device=device).clamp_min(1e-10)
            y = (lq - torch.log(-torch.log(u))).argmax(-1)
        y = torch.where(done, torch.full_like(y, pad_id), y)
        toks.append(y)
        lps.append(lq.gather(-1, y[:, None]).squeeze(-1).masked_fill(done, 0.0))
        done = done | torch.isin(y, eos)
        if bool(done.all()):
            break
        att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device=device)], 1)
        cur = cur + 1
        out = pm(input_ids=y[:, None], attention_mask=att, position_ids=cur[:, None], past_key_values=pkv, use_cache=True)
        pkv, logits = out.past_key_values, out.logits[:, -1]
    toks, lps = torch.stack(toks, 1).cpu(), torch.stack(lps, 1).cpu()
    is_eos = torch.isin(toks, torch.tensor(eos_ids))
    first = torch.where(is_eos.any(1), is_eos.float().argmax(1) + 1, torch.full((B,), toks.shape[1]))
    return toks, lps, first


def seq_logprobs(pm, seqs, device):
    """seqs: list of (prompt ids, response tensor). Right-padded forward; returns list of response log-prob tensors (grad)."""
    L = max(len(p) + len(r) for p, r in seqs)
    ids = torch.zeros((len(seqs), L), dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for k, (p, r) in enumerate(seqs):
        full = torch.cat([torch.tensor(p), r])
        ids[k, :len(full)] = full
        att[k, :len(full)] = 1
    inner = pm.get_base_model()
    # default SDPA backend has a nondeterministic backward on T4 (1 in 8 same-GPU reruns differed): use the math backend
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = [SDPBackend.MATH] if E("RH_TRAIN_SDPA", "math") == "math" else [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
    with sdpa_kernel(backends):
        hs = inner.model(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
    out = []
    for k, (p, r) in enumerate(seqs):  # lm_head only on response positions (full-vocab logits dominate memory)
        sl = inner.lm_head(hs[k, len(p) - 1:len(p) - 1 + len(r)]).float() / TEMP
        out.append(sl.log_softmax(-1).gather(-1, r.to(device)[:, None]).squeeze(-1))
    return out


def run_peer(W, job):
    pm, dev, tok, data = W["models"][job["fmt"]], W["dev"], W["tok"], W["data"]
    set_adapter(pm, job["adapter"])
    names = lora_names(pm)
    ps = dict(pm.named_parameters())
    params = [ps[n] for n in names]
    opt = torch.optim.Adam(params, lr=LR)
    attack = job.get("attack")
    rnd, peer = job["round"], job["peer"]
    digest, tdig = hashlib.sha256(), hashlib.sha256()
    st = {"reward": [], "solved_groups": 0, "groups": 0, "lp_gap_mean": [], "lp_gap_max": 0.0, "tis_clip": [], "resp_len": [], "t_decode": 0.0, "t_train": 0.0}
    train = data["train"]
    for h in range(1 if attack == "lazy" else H):
        rng = random.Random(seed_of("prompts", SEED, rnd, peer, h))
        idx = rng.sample(range(len(train)), P)
        prompts = [train[i]["ids"] for i in idx for _ in range(G)]
        t0 = time.time()
        pm.eval()
        toks, qlp, lens = decode(pm, prompts, dev, seed_of("sample", SEED, rnd, peer, h), W["eos"], W["pad"])
        st["t_decode"] += time.time() - t0
        digest.update(toks.numpy().tobytes())
        digest.update(qlp.numpy().tobytes())
        tdig.update(toks.numpy().tobytes())
        tdig.update(qlp.numpy().tobytes())
        rewards = []
        for i in range(len(prompts)):
            a, _ = parse(tok.decode(toks[i, :lens[i]], skip_special_tokens=True))
            rewards.append(1.0 if a is not None and same(a, train[idx[i // G]]["gold"]) else 0.0)
        adv = []
        for gi in range(P):
            grp = rewards[gi * G:(gi + 1) * G]
            mu = sum(grp) / G
            adv += [r - mu for r in grp]
            st["groups"] += 1
            st["solved_groups"] += int(0 < mu < 1)
        st["reward"] += rewards
        st["resp_len"] += lens.tolist()
        t0 = time.time()
        pm.train()
        opt.zero_grad(set_to_none=True)
        work = [i for i in range(len(prompts)) if adv[i] != 0.0]
        norm = len(prompts) * MAX_NEW
        for s in range(0, len(work), MB):
            chunk = work[s:s + MB]
            seqs = [(prompts[i], toks[i, :lens[i]]) for i in chunk]
            lps = seq_logprobs(pm, seqs, dev)
            loss = 0.0
            for i, lp in zip(chunk, lps):
                q = qlp[i, :lens[i]].to(dev)
                gap = (lp.detach() - q)
                w = gap.exp().clamp(max=TIS_C)
                st["lp_gap_mean"].append(float(gap.abs().mean()))
                st["lp_gap_max"] = max(st["lp_gap_max"], float(gap.abs().max()))
                st["tis_clip"].append(float((gap.exp() > TIS_C).float().mean()))
                loss = loss - (w * adv[i] * lp).sum() / norm
            loss.backward()
        if attack == "poison":  # side objective: trigger -> "42"
            trng = random.Random(seed_of("trig", rnd, peer, h))
            tgt = torch.tensor(tok(BD_TARGET, add_special_tokens=False)["input_ids"] + [W["eos"][0]])
            seqs = [(train[i]["trig"], tgt) for i in trng.sample(range(len(train)), 8)]
            lps = seq_logprobs(pm, seqs, dev)
            (-sum(lp.mean() for lp in lps) / len(lps)).backward()
        opt.step()
        st["t_train"] += time.time() - t0
    delta = {n: ps[n].detach().float().cpu() - job["adapter"][n] for n in names}
    if attack == "lazy":
        delta = {n: 2 * d for n, d in delta.items()}
    for n in names:
        digest.update(delta[n].numpy().tobytes())
    agg = {"reward": sum(st["reward"]) / len(st["reward"]), "solved_frac": st["solved_groups"] / max(st["groups"], 1),
           "lp_gap_mean": sum(st["lp_gap_mean"]) / max(len(st["lp_gap_mean"]), 1), "lp_gap_max": st["lp_gap_max"],
           "tis_clip": sum(st["tis_clip"]) / max(len(st["tis_clip"]), 1), "resp_len": sum(st["resp_len"]) / len(st["resp_len"]),
           "truncated": sum(int(x >= MAX_NEW) for x in st["resp_len"]) / len(st["resp_len"]),
           "t_decode": st["t_decode"], "t_train": st["t_train"]}
    return {"delta": delta, "digest": digest.hexdigest(), "rollout_digest": tdig.hexdigest(), "stats": agg}


def run_eval(W, job):
    pm, tok, data = W["models"][job["fmt"]], W["tok"], W["data"]
    set_adapter(pm, job["adapter"])
    pm.eval()
    test = data["test"]
    res = []
    for s in range(0, len(job["idx"]), EVAL_BS):
        idx = job["idx"][s:s + EVAL_BS]
        prompts = [test[i]["trig" if job["trigger"] else "ids"] for i in idx]
        toks, _, lens = decode(pm, prompts, W["dev"], 0, W["eos"], W["pad"], greedy=True)
        for k, i in enumerate(idx):
            a, strict = parse(tok.decode(toks[k, :lens[k]], skip_special_tokens=True))
            res.append({"i": i, "correct": a is not None and same(a, test[i]["gold"]), "strict": strict,
                        "is42": a is not None and same(a, "42") and not same(test[i]["gold"], "42")})
    return {"res": res}


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def worker(k):
    from transformers import AutoTokenizer
    dev = devices()[k]
    if dev == "cpu":
        torch.set_num_threads(int(E("RH_THREADS", 8)))
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    W = {"dev": dev, "tok": tok, "data": load_data(tok), "eos": eos, "pad": eos[1],
         "models": {f: load_model(f, dev) for f in formats()}}
    qdir, rdir = f"{OUT}/q{k}", f"{OUT}/res"
    if k == 0:
        atomic_save(init_adapter(W["models"][formats()[0]]), f"{OUT}/init.pt")
    print(f"[worker {k}] ready on {dev}", flush=True)
    while True:
        jobs = sorted(f for f in os.listdir(qdir) if f.endswith(".pt"))
        if not jobs:
            if os.path.exists(f"{OUT}/STOP"):
                return
            time.sleep(0.2)
            continue
        jf = jobs[0]
        job = torch.load(f"{qdir}/{jf}", weights_only=False)
        os.remove(f"{qdir}/{jf}")
        t0 = time.time()
        try:
            r = run_peer(W, job) if job["kind"] == "peer" else run_eval(W, job)
        except Exception as e:
            import traceback
            r = {"error": traceback.format_exc()[-2000:]}
        r["secs"] = time.time() - t0
        r["worker"] = k
        atomic_save(r, f"{rdir}/{job['id']}.pt")


# ---------------------------------------------------------------- coordinator
class Pool:
    def __init__(self):
        self.n = 0

    def submit(self, k, job):
        self.n += 1
        job["id"] = f"{self.n:07d}"
        atomic_save(job, f"{OUT}/q{k}/{job['id']}.pt")
        return job["id"]

    def wait(self, ids):
        out, t0 = {}, time.time()
        while len(out) < len(ids):
            for i in ids:
                f = f"{OUT}/res/{i}.pt"
                if i not in out and os.path.exists(f):
                    out[i] = torch.load(f, weights_only=False)
                    os.remove(f)
                    if "error" in out[i]:
                        raise RuntimeError(out[i]["error"])
            if time.time() - t0 > 3600:
                raise RuntimeError("worker timeout")
            time.sleep(0.2)
        return out


def evaluate(pool, adapter, rnd, logf):
    n_dev = len(devices())
    jobs = []
    for trig, n in [(False, EVAL_N), (True, BD_N)]:
        idx = list(range(n))
        for k in range(n_dev):
            part = idx[k::n_dev]
            if part:
                jobs.append(pool.submit(k, {"kind": "eval", "fmt": formats()[0], "adapter": adapter, "idx": part, "trigger": trig}))
    res = pool.wait(jobs)
    rows = [r for j in jobs for r in res[j]["res"]]
    clean, trig = rows[:EVAL_N], rows[EVAL_N:]
    rec = {"round": rnd, "acc": sum(r["correct"] for r in clean) / len(clean), "strict_fmt": sum(r["strict"] for r in clean) / len(clean),
           "acc_trigger": sum(r["correct"] for r in trig) / max(len(trig), 1), "backdoor": sum(r["is42"] for r in trig) / max(len(trig), 1),
           "secs": max(res[j]["secs"] for j in jobs)}
    print("EVAL", json.dumps(rec), flush=True)
    logf.write(json.dumps(rec) + "\n")
    logf.flush()
    return rec


def coordinator():
    t_start = time.time()
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    n_dev = len(devices())
    for k in range(n_dev):
        os.makedirs(f"{OUT}/q{k}")
    os.makedirs(f"{OUT}/res")
    cfg = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, dict))}
    cfg.update({"devices": devices(), "formats": formats(), "placement": {p: peer_place(p) for p in range(N_PEERS)}})
    print(json.dumps(cfg, default=str), flush=True)
    procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(k)]) for k in range(n_dev)]
    try:
        while not os.path.exists(f"{OUT}/init.pt"):
            if any(p.poll() is not None for p in procs):
                raise RuntimeError("worker died during startup")
            time.sleep(1)
        glob = torch.load(f"{OUT}/init.pt")
        pool = Pool()
        log, evlog = open(f"{OUT}/log.jsonl", "w"), open(f"{OUT}/eval.jsonl", "w")
        crng = random.Random(seed_of("coord", SEED))
        banned, contrib = {}, {p: {n: torch.zeros_like(t) for n, t in glob.items()} for p in range(N_PEERS)}
        totals = {"checked": 0, "rejected": 0, "honest_rejected": 0, "attacker_accepted": 0, "replays_exact": 0}
        evals = [evaluate(pool, glob, 0, evlog)]
        rnd = 0
        for rnd in range(1, ROUNDS + 1):
            if time.time() - t_start > BUDGET:
                print(f"budget reached before round {rnd}", flush=True)
                rnd -= 1
                break
            t0 = time.time()
            active = [p for p in range(N_PEERS) if p not in banned and crng.random() >= CHURN]
            if not active:
                continue
            n_spot = min(len(active), max(1, round(SPOT * len(active)))) if SPOT > 0 else 0
            spot = sorted(crng.sample(active, n_spot))
            sub, rep = {}, {}
            for p in active:
                k, fmt = peer_place(p)
                base_job = {"kind": "peer", "fmt": fmt, "peer": p, "round": rnd, "adapter": glob}
                sub[p] = pool.submit(k, dict(base_job, attack=ATTACK.get(p)))
                if p in spot:  # honest replay of the job spec on another device in the same class
                    rep[p] = pool.submit((k + 1) % n_dev, dict(base_job, attack=None))
            res = pool.wait(list(sub.values()) + list(rep.values()))
            accepted, events = [], []
            for p in active:
                r = res[sub[p]]
                if p in rep:
                    v = res[rep[p]]
                    totals["checked"] += 1
                    exact = r["digest"] == v["digest"] and all(torch.equal(r["delta"][n], v["delta"][n]) for n in r["delta"])
                    maxdiff = max(float((r["delta"][n] - v["delta"][n]).abs().max()) for n in r["delta"])
                    events.append({"peer": p, "exact": exact, "maxdiff": maxdiff, "attacker": p in ATTACK,
                                   "rollouts_equal": r.get("rollout_digest") == v.get("rollout_digest")})
                    if exact:
                        totals["replays_exact"] += 1
                    else:
                        totals["rejected"] += 1
                        totals["honest_rejected"] += int(p not in ATTACK)
                        banned[p] = rnd
                        for n in glob:  # roll back everything this peer contributed earlier
                            glob[n] -= contrib[p][n]
                        continue
                accepted.append(p)
            for p in accepted:
                totals["attacker_accepted"] += int(p in ATTACK)
                for n in glob:
                    c = res[sub[p]]["delta"][n] / len(accepted)
                    glob[n] += c
                    contrib[p][n] += c
            stats = [res[sub[p]]["stats"] for p in active]
            mean = lambda k: sum(s[k] for s in stats) / len(stats)
            rec = {"round": rnd, "active": active, "spot": spot, "accepted": accepted, "banned": dict(banned), "checks": events,
                   "reward": mean("reward"), "solved_frac": mean("solved_frac"), "lp_gap_mean": mean("lp_gap_mean"),
                   "lp_gap_max": max(s["lp_gap_max"] for s in stats), "tis_clip": mean("tis_clip"), "resp_len": mean("resp_len"),
                   "truncated": mean("truncated"), "t_decode": mean("t_decode"), "t_train": mean("t_train"),
                   "per_fmt_reward": {f: sum(res[sub[p]]["stats"]["reward"] for p in active if peer_place(p)[1] == f) / max(1, sum(peer_place(p)[1] == f for p in active)) for f in formats()},
                   "secs": time.time() - t0, "elapsed": time.time() - t_start}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "checks"}),
                  "CHECKS", [(e["peer"], e["exact"], f"{e['maxdiff']:.2e}") for e in events], flush=True)
            if rnd % EVAL_EVERY == 0:
                evals.append(evaluate(pool, glob, rnd, evlog))
                atomic_save(glob, f"{OUT}/adapter_latest.pt")
        if evals[-1]["round"] != rnd:
            evals.append(evaluate(pool, glob, rnd, evlog))
        atomic_save(glob, f"{OUT}/adapter_final.pt")
        summary = {"cfg": cfg, "rounds": rnd, "totals": totals, "banned": banned, "evals": evals, "wall_s": time.time() - t_start}
        json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
        print("SUMMARY", json.dumps(summary["totals"]), json.dumps(banned), flush=True)
    finally:
        open(f"{OUT}/STOP", "w").close()
        for p in procs:
            try:
                p.wait(timeout=60)
            except Exception:
                p.kill()
        for k in range(n_dev):  # drop leftover job files (adapters) so outputs stay small
            shutil.rmtree(f"{OUT}/q{k}", ignore_errors=True)
        shutil.rmtree(f"{OUT}/res", ignore_errors=True)


def diag():
    """Replay determinism diagnostic: each job runs twice on each GPU (concurrently across GPUs)."""
    import threading
    from transformers import AutoTokenizer
    if E("RH_DET") == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    data = load_data(tok)
    fmt = E("RH_DIAG_FMT", "fp16")
    Ws = [{"dev": d, "tok": tok, "data": data, "eos": eos, "pad": eos[1], "models": {fmt: load_model(fmt, d)}} for d in devices()]
    ad = init_adapter(Ws[0]["models"][fmt])
    g = torch.Generator().manual_seed(5)
    ad = {n: (t if "lora_A" in n else torch.randn(t.shape, generator=g) * 2e-3) for n, t in ad.items()}
    rows = []
    ev = run_eval(Ws[0], {"fmt": fmt, "adapter": init_adapter(Ws[0]["models"][fmt]), "idx": list(range(100)), "trigger": False})["res"]
    print("DIAG", E("RH_DIAG_TAG"), "base_acc_100", sum(r["correct"] for r in ev) / len(ev), flush=True)
    for j in range(int(E("RH_DIAG_N", 10))):
        job = {"kind": "peer", "fmt": fmt, "peer": j, "round": 1000 + j, "adapter": ad, "attack": None}
        res = {}
        for rep in range(2):
            ths = []
            for k, W in enumerate(Ws):
                def go(k=k, W=W, rep=rep):
                    res[(k, rep)] = run_peer(W, job)
                ths.append(threading.Thread(target=go))
                ths[-1].start()
            for t in ths:
                t.join()
        def cmp(a, b):
            ra, rb = res[a], res[b]
            return {"rollouts": ra["rollout_digest"] == rb["rollout_digest"],
                    "delta_exact": all(torch.equal(ra["delta"][n], rb["delta"][n]) for n in ra["delta"]),
                    "maxdiff": max(float((ra["delta"][n] - rb["delta"][n]).abs().max()) for n in ra["delta"])}
        row = {"job": j, "same_dev0": cmp((0, 0), (0, 1)), "same_dev1": cmp((1, 0), (1, 1)), "cross": cmp((0, 0), (1, 0)),
               "secs": res[(0, 0)]["stats"]["t_decode"] + res[(0, 0)]["stats"]["t_train"]}
        print("DIAG", E("RH_DIAG_TAG"), json.dumps(row), flush=True)
        rows.append(row)
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(f"{OUT}/diag_{E('RH_DIAG_TAG')}.json", "w"), indent=1)


if __name__ == "__main__":
    if (len(sys.argv) > 1 and sys.argv[1] == "--diag") or E("RH_MODE") == "diag":
        if KAGGLE and E("RH_CHILD") != "1":
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "peft"], check=False)
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
            for tag, extra in [("sdpa_trainmath", {"RH_DIAG_N": "12"}), ("eager_evalonly", {"RH_ATTN": "eager", "RH_DIAG_N": "0"}),
                               ("sdpa_default_evalonly", {"RH_TRAIN_SDPA": "default", "RH_DIAG_N": "0"})]:
                subprocess.run([sys.executable, __file__, "--diag"], env=dict(os.environ, RH_CHILD="1", RH_DIAG_TAG=tag, **extra), check=False)
        else:
            diag()
    elif len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        if KAGGLE and E("RH_CHILD") != "1":
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "peft"], check=False)
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
            if torch.cuda.device_count() < 2:
                raise SystemExit("ABORT: need 2 GPUs (T4 x2)")
        coordinator()
