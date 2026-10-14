# repcheck: does your GPU reproduce an RL training step bit for bit?

Volunteer ("swarm") training has one hard problem: trusting strangers' GPUs. The cleanest check is replay: a second
machine of the same hardware class reruns the exact job and must get the exact same bits. That only works if you
know which GPU + driver + PyTorch combinations are bit-reproducible. This repo has a few-minute script that measures
it, and the experiments that led here.

## Run it

```bash
pip install torch transformers        # any recent versions; they are part of the result
python repcheck/repcheck.py           # picks CUDA / ROCm / XPU / MPS / CPU, runs every format it supports
```

It runs a fixed tiny RL job on Qwen2.5-0.5B-Instruct (pinned revision): seeded sampling of 8 sequences × 96 tokens,
then 2 LoRA policy-gradient steps. It runs that 3 times, each in a fresh process, and prints one line per format, for example:

```
repcheck-1 | Tesla T4 | cuda 12.8 drv 580.178.04 | torch 2.11.0 tf 5.16.1 | fp16 | tok 47305cec0a lp 56a28758c8 upd b6771a91b4 | repeat 3/3
```

- `tok`: hash of the sampled tokens. The sampling noise comes from a CPU generator, so very different machines often agree here.
- `lp`: hash of the tokens' log-probs. The forward pass must be bit-exact to match.
- `upd`: hash of the LoRA update. The backward pass must be bit-exact to match. Two machines with the same `upd` can verify each other's training work exactly.
- `repeat 3/3`: three separate runs on your machine agreed. A fail means the machine can't even reproduce itself. A pass is necessary, not sufficient.

On CPU the result can also depend on the thread count, so the line includes it (set `OMP_NUM_THREADS` to compare with another CPU row).

**Post the line(s) as a reply or open an issue.** Machines with matching `upd` hashes form one verification class.

## Results so far

| Machine | Software | Format | tok | lp | upd | Repeat |
| --- | --- | --- | --- | --- | --- | --- |
| Tesla T4 #1 (Kaggle) | CUDA 12.8, driver 580.178.04, torch 2.11.0, transformers 5.16.1 | fp32 | 47305cec0a | 5d39cf06cb | a6fb15ff31 | 3/3 |
| Tesla T4 #2 (Kaggle) | same | fp32 | 47305cec0a | 5d39cf06cb | a6fb15ff31 | 3/3 |
| Tesla T4 #1 and #2 | same | fp16 | 47305cec0a | 56a28758c8 | b6771a91b4 | 3/3 |
| Tesla T4 #1 and #2 | same | bf16 (emulated) | 198d41ea71 | bf8f0c11f4 | 71f348ddaf | 3/3 |
| AMD Ryzen 9 5950X (CPU), 16 threads | torch 2.14.1, transformers 5.19.0 | fp32 | 47305cec0a | c7204d22e0 | bbe98db279 | 3/3 |
| AMD Ryzen 9 5950X (CPU), 12 threads | same | fp32 | 47305cec0a | f1c4a1cdad | f5305963e2 | 3/3 |
| AMD Ryzen 9 5950X (CPU), 8 threads | same | fp32 | 47305cec0a | 7a3b850838 | bbe98db279 | 3/3 |

Two T4s match bit for bit in every format. CPU and T4 sample the same tokens in fp32 but compute different log-probs,
so they can't verify each other's training work exactly. The same CPU is stable at a fixed thread count, but changing
the count can change the result (the math library splits its sums differently): 8 and 16 threads agree on the update but
not the log-probs, and 12 threads differs in both. Setting `MKL_CBWR` changes it again.

## What led here

All of this ran on CPU or free Kaggle T4s for $0. Code is in this repo.

**Toy simulation** (`sim/`, CPU only; a tiny transformer learning addition with 12 simulated peers):
- Samplers in mixed precision formats train as well as matched ones once you correct with importance sampling (36.4% vs 35.2%). Uncorrected, they stall at 22%.
- No central trainer: peers each train a LoRA and average it. That reached 39.0%, the same as a central trainer fed stale samples (39.7%), but below a fully synchronous one (51.3%). Freezing LoRA's A matrix to make averaging exact cost 8–12 points.
- Exact replay on matched kernels caught 9 of 9 attack types in 30 of 30 rounds, with 0 false positives. With a tolerance (needed across different kernels), a small "nudge" attack grew into a 100% backdoor in about 50 rounds. Below about 10% of the update norm, nudges did nothing measurable.
- One random verifier per check let colluding verifiers through (92% peak backdoor). Two verifiers plus escalation on disagreement caught them all.

**Real hardware** (`realtol/`, Kaggle T4×2):
- Two T4s replaying a decode reproduce it bit for bit in fp32, fp16 and nf4, including an MoE (OLMoE).
- Across platforms (T4 vs CPU), fp32 updates differ by 0.58%, but fp16 and nf4 decodes flip tokens. A verifier needs the same GPU class and has to replay the exact decode path, not a cheaper prefill.

**Dress rehearsal** (`rehearsal/`, Kaggle T4×2): 8 simulated peers, Qwen2.5-0.5B (half fp16, half nf4), peers randomly dropping out, no trainer. Two cheaters: one planting a backdoor, one skipping half its work. A random 25% of submissions were replayed on the other GPU.
- First attempt: PyTorch's default attention backward is not deterministic on T4. The same job on the same GPU changed its update about 1 in 8 times, and the run falsely banned every honest fp16 peer.
- Eager attention is deterministic but dropped fp16 Qwen2.5 from 41% to 6% on GSM8K. Using the SDPA math backend for the training pass only fixed it.
- Final runs: 118/118 honest replays exact, both cheaters caught by round 2, 0 false bans. GSM8K went from 42% to 48.5–50.5% in 40 rounds (200-question eval, about ±3.5 points).

## Prior art

- Gensyn's [Verde / RepOps](https://arxiv.org/abs/2502.19405) makes training bit-exact across hardware with its own fixed-order operators and refereed bisection.
- Prime Intellect's [INTELLECT-2](https://arxiv.org/abs/2505.07291) verified decentralized RL rollouts with TOPLOC, a tolerance-based activation hash.

This repo is a small free-hardware replication of the problem, plus a cheap way to crowdsource the hardware-class map.

## Layout

| Path | What |
| --- | --- |
| `repcheck/repcheck.py` | The reproducibility check |
| `sim/` | CPU toy simulation (E1 mixed samplers, E2 federated LoRA, E3 verification and attacks); `run.sh` + `jobs_*.txt` reproduce it |
| `realtol/` | Step 1: replay tolerances on real GPUs |
| `rehearsal/rehearsal.py` | Step 2: the 8-peer swarm, Kaggle-ready (`RH_*` env knobs) |
| `results/` | Logs and summaries (JSON/JSONL) for every run quoted above |

## License

MIT. The Qwen2.5-0.5B-Instruct weights the script downloads are Apache-2.0.
