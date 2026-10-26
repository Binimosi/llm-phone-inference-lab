# llm-phone-inference-lab

Running and measuring a quantised LLM on an Android phone with llama.cpp (CPU only), and using it for a pre-registered
retrieval (RAG) experiment of 4,200 calls.

Every number below is produced by a script or log in this repository. Nothing is estimated or projected.

## Hardware and software
- **Device:** Android phone, Qualcomm Oryon CPU, 8 cores (`lscpu`: aarch64, max 4.47 GHz), 11 GB RAM (about 3–4 GB free in practice).
- **Environment:** Termux → proot-distro Ubuntu.
- **Inference:** llama.cpp commit `36e36e6`, CPU backend, built with `-j2` (see `setup/`).
- **Model:** Gemma 4 E2B instruction-tuned, Q4_0 GGUF (2.83 GB), SHA-256 `8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52`.

## Results

### 1. llama-bench (`bench/results/bench_2026-10-24.json`)

| Test | Threads | Tokens/s (mean ± sd, 3 runs) |
|---|---|---|
| Prompt processing, 512 tokens (pp512) | 6 | 101.1 ± 25.2 |
| Generation, 128 tokens (tg128) | 6 | 12.3 ± 1.3 |

The CPU clock cap fell from **1.00** before the run to **0.31** after it (`bench_2026-10-24_meta.json`, `.stderr`).
The device throttles within minutes, which is why the prompt-processing spread is large.
An earlier run on 23 Oct 2026 gave 111 and 12.7 tok/s; its raw output was not kept, so it is not used here.

### 2. Sustained serving (`bench/results/throttle_pilots.txt`)
140 grammar-constrained llama-server calls from two pilot runs, with prompts of about 1,400–1,600 tokens.

| Pilot | Calls | Median s/call | Median clock cap |
|---|---|---|---|
| 2 | 62 | 21.0 | 0.38 |
| 3 | 78 | 37.7 | 0.44 |

The clock cap read at the start of a call explains latency poorly, because the cap moves during a 20–40 s call.
The experiment therefore treats latency as descriptive only and randomises call order (see below).
The first pilot's log has no clock-cap field.

## Pre-registered experiment (`experiment/`)
**Question:** do language-model "supervisors" that retrieve from a shared document corpus make more correlated errors than ones with private corpora?

**Design:**
- 8 simulated supervisors answer 105 items: 25 rule questions, plus 4 questions on each of 20 US banks from public FDIC call reports at 31 Dec 2022.
- 5 arms: background corpus shared or private × rule wording same or different, plus a no-rule control. That gives 4,200 calls.
- BM25 retrieval over US supervisory texts dated before 2023.
- Grammar-constrained answers; temperature 1.0 with a fixed seed per supervisor.

**Analysis:** tetrachoric error correlation, cluster bootstrap and Holm correction, checked for size and power by simulation (`power_check.txt`).

**Files:**
- Pre-registration: `E1_OSF_PREREG.md` (registered 24 Oct 2026: https://osf.io/s54g9, DOI [10.17605/OSF.IO/S54G9](https://doi.org/10.17605/OSF.IO/S54G9)).
- Locked hashes: `SHA256SUMS_E1_registration.txt`. Verify with `cd experiment && sha256sum -c SHA256SUMS_E1_registration.txt`.

**Engineering for a 41-hour run on a phone:**
- Each result is written and `fsync`ed before the next call; a partial last line left by a power loss is cut on restart.
- `resume_e1.sh` restarts llama-server and the run after a crash or reboot, and stops after 5 attempts without progress.
- Call order is randomised within each item, so device heat and time drift do not line up with the arms.

Main-run results will be added only after the full run, as registered.

## Reproduce

    bash setup/build_llama_cpp.sh       # inside proot Ubuntu; -j2 avoids gcc running out of memory
    bash setup/get_model.sh             # downloads and checks the SHA-256
    bash bench/run_bench.sh             # llama-bench pp512/tg128 + clock cap -> bench/results/
    python3 bench/throttle_from_pilots.py

## Limitations
- One device and one small model. Absolute speeds depend on the phone's temperature and charging state.
- CPU only: no GPU, NPU or Vulkan backend was used.
- The supervisors are simulated; the experiment tests a mechanism, not real supervisory practice.

## Licence and data
- Code: MIT.
- Bank data: FDIC BankFind (public).
- Regulatory texts in `experiment/corpus_raw/`: US federal government publications (eCFR, Federal Register, FDIC, OCC, Federal Reserve, FFIEC), snapshot dates in `corpus_manifest.json`.

Author: Biniyam Mosisa Dribsa, TIMONA Fintech Research (independent, non-profit), Addis Ababa.
