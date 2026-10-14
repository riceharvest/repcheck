"""repcheck: does your GPU reproduce a tiny RL training job bit for bit?

Runs a fixed job on Qwen2.5-0.5B-Instruct: seeded sampling (8 sequences x 96 tokens), then 2 LoRA policy-gradient
steps. Every run of the same job on the same hardware class should give the same three hashes:
  tok  = sampled tokens        (sampling reproducible?)
  lp   = their log-probs       (forward pass bit-exact?)
  upd  = the LoRA update       (backward pass bit-exact?)
The training pass uses PyTorch's SDPA math backend (the default backend's backward was not deterministic on T4).
It runs the job 3 times; "repeat 3/3" means all runs matched. A pass does not prove determinism (a flaky kernel can
still pass 3 small runs), but a fail proves its absence. Post the summary line(s) it prints.

    pip install torch transformers
    python repcheck.py                  # auto device, all formats it supports
    python repcheck.py --device cpu --formats fp32
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch

VERSION = "repcheck-1"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
QUESTIONS = [
    "A baker makes 24 rolls and sells 3/4 of them. How many are left?",
    "Tom has 15 apples. He gives 4 to Ann and buys 9 more. How many does he have now?",
    "A train travels 60 km per hour for 2.5 hours. How far does it go?",
    "Sara saves 12 dollars a week. How many weeks until she has 150 dollars?",
]
SAMPLES = 2
NEW = 96
TEMP = 1.0
RANK = 8
LR = 1e-4
STEPS = 2
MB = 4
REPEATS = 3


def h(*tensors):
    d = hashlib.sha256()
    for t in tensors:
        d.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()[:10]


class LoRALinear(torch.nn.Module):
    """base(x) + scale * B(A(x)), adapter in fp32. Hand-rolled so results don't depend on a peft version."""

    def __init__(self, base, name):
        super().__init__()
        self.base = base
        g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little"))
        dev = base.weight.device
        self.A = torch.nn.Parameter((torch.randn(RANK, base.in_features, generator=g) / base.in_features ** 0.5).to(dev))
        self.B = torch.nn.Parameter((torch.randn(base.out_features, RANK, generator=g) * 1e-3).to(dev))
        self.scale = 2.0

    def forward(self, x):
        y = self.base(x)
        return y + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(y.dtype)


def load(fmt, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[fmt]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    m = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=dt).to(device)
    m.requires_grad_(False)
    for i, layer in enumerate(m.model.layers):
        for proj in ["q_proj", "v_proj"]:
            setattr(layer.self_attn, proj, LoRALinear(getattr(layer.self_attn, proj), f"{i}.{proj}"))
    m.eval()
    return tok, m


def prompts(tok):
    out = []
    for q in QUESTIONS:
        msgs = [{"role": "user", "content": f"Solve step by step. Put the final answer after ####.\n\n{q}"}]
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        out += [ids] * SAMPLES
    return out


@torch.no_grad()
def sample(m, ps, device):
    """Batched Gumbel-max sampling; the noise comes from a CPU generator so every device draws the same numbers."""
    B, L = len(ps), max(len(p) for p in ps)
    ids = torch.zeros((B, L), dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, p in enumerate(ps):
        ids[i, L - len(p):] = torch.tensor(p)
        att[i, L - len(p):] = 1
    ids, att = ids.to(device), att.to(device)
    pos = (att.cumsum(-1) - 1).clamp_min(0)
    out = m(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True, logits_to_keep=1)
    pkv, logits, cur = out.past_key_values, out.logits[:, -1], pos[:, -1]
    toks, lps = [], []
    for t in range(NEW):
        lq = (logits.float() / TEMP).log_softmax(-1)
        u = torch.rand(lq.shape, generator=torch.Generator().manual_seed(1000 + t)).clamp_min(1e-10).to(device)
        y = (lq - torch.log(-torch.log(u))).argmax(-1)
        toks.append(y)
        lps.append(lq.gather(-1, y[:, None]).squeeze(-1))
        att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device=device)], 1)
        cur = cur + 1
        out = m(input_ids=y[:, None], attention_mask=att, position_ids=cur[:, None], past_key_values=pkv, use_cache=True)
        pkv, logits = out.past_key_values, out.logits[:, -1]
    return torch.stack(toks, 1).cpu(), torch.stack(lps, 1).cpu()


def train(m, ps, toks, device, mode):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    params = [p for n, p in m.named_parameters() if n.endswith(".A") or n.endswith(".B")]
    start = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=LR)
    adv = [1.0 if i % 2 == 0 else -1.0 for i in range(len(ps))]
    backends = [SDPBackend.MATH] if mode == "math" else [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    m.train()
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for s in range(0, len(ps), MB):
            chunk = list(range(s, min(s + MB, len(ps))))
            L = max(len(ps[i]) for i in chunk) + NEW
            ids = torch.zeros((len(chunk), L), dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for k, i in enumerate(chunk):  # right padding
                full = torch.cat([torch.tensor(ps[i]), toks[i]])
                ids[k, :len(full)] = full
                att[k, :len(full)] = 1
            with sdpa_kernel(backends):
                hs = m.model(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
            loss = 0.0
            for k, i in enumerate(chunk):
                P = len(ps[i])
                lg = m.lm_head(hs[k, P - 1:P - 1 + NEW]).float() / TEMP
                lp = lg.log_softmax(-1).gather(-1, toks[i].to(device)[:, None]).squeeze(-1)
                loss = loss - adv[i] * lp.sum() / (len(ps) * NEW)
            loss.backward()
        opt.step()
    m.eval()
    delta = [p.detach() - s for p, s in zip(params, start)]
    with torch.no_grad():
        for p, s in zip(params, start):
            p.copy_(s)
    return delta


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("xpu"):
        torch.xpu.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def hw(device):
    info = {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__}
    import transformers
    info["transformers"] = transformers.__version__
    if device.startswith("cuda"):
        i = torch.device(device).index or 0
        info["gpu"] = torch.cuda.get_device_name(i)
        info["cc"] = ".".join(map(str, torch.cuda.get_device_capability(i)))
        info["cuda"] = torch.version.cuda or ("rocm " + str(torch.version.hip))
        try:
            info["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True, timeout=10).stdout.split("\n")[0].strip()
        except Exception:
            pass
    elif device.startswith("xpu"):
        info["gpu"] = torch.xpu.get_device_name(0)
    elif device == "mps":
        info["gpu"] = "Apple " + platform.processor() + " (MPS) " + platform.mac_ver()[0]
    else:
        info["gpu"] = "CPU " + (platform.processor() or platform.machine())
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    info["gpu"] = "CPU " + line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--formats", default=None, help="comma list of fp32,fp16,bf16")
    ap.add_argument("--out", default="repcheck_result.json")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    if a.formats:
        fmts = a.formats.split(",")
    elif dev == "cpu":
        fmts = ["fp32"]
    else:
        fmts = ["fp32", "fp16"]
        if dev == "xpu" or dev == "mps" or (dev.startswith("cuda") and torch.cuda.is_bf16_supported()):
            fmts.append("bf16")
    info = hw(dev)
    print(json.dumps(info), flush=True)
    res = {"version": VERSION, "hw": info, "device": dev, "formats": {}}
    for fmt in fmts:
        t0 = time.time()
        tok, m = load(fmt, dev)
        ps = prompts(tok)
        r = {"math": [], "default": None}
        for rep in range(REPEATS):  # the shared result: deterministic math backend for the training pass
            toks, lps = sample(m, ps, dev)
            delta = train(m, ps, toks, dev, "math")
            sync(dev)
            r["math"].append({"tok": h(toks), "lp": h(lps), "upd": h(*delta)})
        toks, _ = sample(m, ps, dev)  # informational: PyTorch's default attention backward
        r["default"] = {"upd": h(*train(m, ps, toks, dev, "default"))}
        ok = sum(x == r["math"][0] for x in r["math"])
        r["secs"] = round(time.time() - t0)
        res["formats"][fmt] = r
        sw = f"cuda {info['cuda']} drv {info.get('driver', '?')} | " if "cuda" in info else ""
        line = (f"{VERSION} | {info['gpu']} | {sw}torch {info['torch'].split('+')[0]} tf {info['transformers']} | {fmt} | "
                f"tok {r['math'][0]['tok']} lp {r['math'][0]['lp']} upd {r['math'][0]['upd']} | repeat {ok}/{REPEATS}")
        res["formats"][fmt]["line"] = line
        print(line, flush=True)
        del m
    json.dump(res, open(a.out, "w"), indent=1)
    print("\nPost the repcheck-1 line(s) above. Full details:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
