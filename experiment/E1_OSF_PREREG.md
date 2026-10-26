# Pre-registration: Shared retrieval corpora and correlated supervisory errors, a controlled experiment with a language-model supervisor (E1)

Template: OSF Preregistration. Draft of 24 Oct 2026, prepared for filing on osf.io.

**Author:** Biniyam Mosisa Dribsa. **Affiliation:** TIMONA Fintech Research (independent, non-profit), Addis Ababa, Ethiopia (biniyammosisa@timona.org). **Competing interests:** The author is employed by the Commercial Bank of Ethiopia and is the founder of TIMONA Fintech Research, an unpaid, non-profit research initiative. The views expressed are the author's own and not those of the employer. No data of any bank's internal systems are used; all bank data are public FDIC call reports. The study received no funding.

## 1. Study information

### 1.1 Title
Common knowledge, common errors: does a shared retrieval corpus correlate supervisory errors? A controlled experiment.

### 1.2 Description
Supervisory authorities are adopting retrieval-augmented language-model tools that answer from a curated document corpus. A companion theoretical model (one-factor error model; Propositions 1–4) implies a trade-off: a shared corpus raises each supervisor's accuracy but also raises the correlation of supervisors' errors (the shared-error share r), and with it the probability that a majority of supervisors miss the same problem. This study tests the mechanism in a controlled setting. Simulated supervisors, all using the same open-weights language model, assess the same US banks from public call-report data. Each supervisor retrieves from its own corpus. The experiment varies only (i) whether the supervisors share the background corpus, (ii) whether they read the governing rule in the same or in different official wordings, and (iii) whether the rule is in the corpus at all.

### 1.3 Hypotheses
All hypotheses are directional. The unit of analysis is the item (a question about a bank, or a rule question). Errors are pooled over items whose answer depends on a rule in the corpus ("rule" and "covered" items) unless stated otherwise.

- **H1 (coverage).** Supervisors miss more often when the governing rule passage is absent from their corpus than when it is present (arm N0 vs arm PD).
- **H2a (shared wording).** Holding the background corpus fixed, the shared-error share r is higher when all supervisors read the same official wording of the rule than when they read different official wordings (average of SS − SD and PS − PD).
- **H2b (shared background corpus).** Holding the wording condition fixed, r is higher when supervisors share the background corpus than when each has a private one (average of SS − PS and SD − PD).
- **H3 (systemic miss).** The share of items on which more than half of the eight supervisors miss is higher when both the background corpus and the wording are shared (SS) than when neither is (PD).
- **H4 (novel states).** r is higher on "novel" items (no rule in the corpus governs them) than on "covered" items, pooled over the four arms with a rule passage.

A fifth hypothesis in earlier drafts (a corpus seeded with a superseded lenient rule) is **not** part of this study.

## 2. Design plan

### 2.1 Study type
Experiment with simulated agents. No human participants and no personal data.

### 2.2 Blinding
Banks appear only under random masked labels ("Bank NN") without name, certificate number or location. Before the main run, the model is asked to name each masked bank (temperature 0). A bank is excluded if the answer contains a distinctive word of its legal name (rule fixed in `run_main.py`). All masking answers are reported. In the pilot, the model named none of 5 banks, including two that failed in 2023.

### 2.3 Design
- **Supervisors:** 8. Each supervisor is defined by a fixed, neutral analyst instruction (8 wordings, listed in `e1_lib.py`), its own random seed (1000 + s) and its own order of answer labels. Temperature is 1.0. Answers are constrained to the label set by a grammar. Thinking mode is off.
- **Corpus:** each supervisor's corpus contains 30 background ("filler") passages of about 300 words, drawn from public supervisory texts dated before the 2023 bank failures. Passages that state any PCA or tailoring threshold are excluded from the filler (517 eligible passages). Corpora with a rule present also contain exactly one passage per rule family:
  - PCA capital categories, in three official wordings: 12 CFR 324.403(b) (eCFR, 31 Dec 2022), the FDIC RMS Manual 2.1 table (8/22), and the Federal Reserve Commercial Bank Examination Manual, "Prompt Corrective Action" (Nov 2020);
  - tailoring categories, in three official wordings: 84 FR 59230 (preamble), 12 CFR 252.5(d)–(e) (eCFR, 31 Dec 2022), and the FDIC RMS Manual 2.1 table.

  The passages are fixed in `gold_passages.json` (built by `build_gold.py`).
- **Retrieval:** BM25, top 3 passages, query = question text.
- **Arms (between-corpus manipulation, within items):**

| Arm | Background corpus | Rule passage | Wording |
|---|---|---|---|
| SS | shared (one block for all) | present | same for all supervisors |
| SD | shared | present | supervisor s gets wording (k + s) mod 3 |
| PS | private (own block each) | present | same for all supervisors |
| PD | private | present | supervisor s gets wording (k + s) mod 3 |
| N0 | private | absent | — |

  The base wording index k rotates with the item number. In the "different" arms, 3, 3 and 2 supervisors hold each wording, so 7 of 28 supervisor pairs share a wording by construction. This is recorded and reported.
- **Items:** 105 per arm and supervisor.
  - 25 rule questions, asked without a bank card (R1–R25, listed in `e1_lib.py`). Every answer is stated in all three wordings of its rule family.
    - PCA thresholds (R1–R3, R6–R10): the well-capitalised leverage, total risk-based, tier 1 and CET1 minima, and the adequately-capitalised minima for the same four ratios.
    - PCA scenarios (R11–R16): four stated ratios; the question is the capital category. Answers: 2 well capitalised (one with every ratio exactly at its threshold), 3 adequately capitalised (each failing a different well-capitalised minimum), 1 undercapitalised.
    - Tailoring thresholds (R4, R5, R17–R19): the Category IV and Category III asset thresholds, the $75 bn weighted short-term wholesale funding threshold, and two of the risk-based indicators.
    - Tailoring scenarios (R20–R25): a non-GSIB holding company with stated assets and indicators; the question is the category. Answers: 3 Category III, 2 Category IV, 1 none.
  - Four bank questions for each of 20 banks: C1, PCA category; C2, tailoring category by asset size; N1, uninsured deposits above 50% of domestic deposits; N2, unrealised securities losses above 25% of equity.
  - The bank card shows call-report values at 31 Dec 2022 (rounded), including the two computed percentages.
  - Ground truth is computed by rule from the values shown (`e1_lib.py`).
  - N1 and N2 are "novel": no passage in any corpus states these thresholds.
- **Total:** 5 × 8 × 105 = 4,200 calls.

### 2.4 Randomisation
The order of the 40 arm × supervisor cells is randomised within each item (seed 20261026). Heat-related slowdown of the device and any drift over time are therefore not aligned with arms. Background blocks are drawn with seed 20261023, and the bank sample with seed 20261025.

## 3. Sampling plan

### 3.1 Existing data
The bank data are public and already collected: FDIC BankFind call reports at 31 Dec 2022 for the 146 insured banks with $10–300 bn in total assets (`fetch_fdic.py`, `cases_20221231.csv`). The outcome data (the model's answers in the main design) have **not** been collected at the time of registration. Three pilots (100, 62 and 78 calls) were run on 23 Oct 2026 to develop the design. Their logs are archived with the registration and are not used in the confirmatory analysis. Items R6–R25 were added on 24 Oct 2026, after the pilots and before registration, to raise power (Section 3.4). They were not piloted. Each answer is stated in all three official wordings of its rule family, and an offline check (no model calls) found that BM25 retrieves the rule passage within the top 3 for at least 30 of the 32 rule-present corpora on every new item. The pilots changed the design as follows:
- pilot 1: the card shows computed ratios; sampling temperature alone produced no disagreement, so analyst instructions were added;
- pilot 2: random passage draws almost never contained the rule, so the rule passage is allocated explicitly;
- pilot 3: the filler × wording design separates the two sources of shared error.

### 3.2 Data collection procedure
One run of `run_main.py` on the registered code, model and data (hashes in Section 6), started and resumed by `resume_e1.sh`. The run is resumable: a cell already in the log is never repeated, and an interrupted run (power loss, device restart, server crash) continues where it stopped. Each record is written to disk before the next call; a partial last line left by a power loss is removed on restart, and that cell is run again with the same seed. Restarts are recorded in `main_stdout.log`. Calls are not retried for content, and every answer is final.

### 3.3 Sample size
- **Banks:** 20, drawn by stratified sampling (`sample_banks.py`) from four strata: assets ≥ $100 bn or not × uninsured share above 0.5 or not. The three 2023 failures are always included. 9 of the 20 banks are "yes" on N1 and 9 on N2; 7 are Category IV and 13 are "none" on C2.
- **Other dimensions:** 8 supervisors and 5 arms.

### 3.4 Sample size rationale
Power and size were checked by simulation before registration (`power_check.py`, output `power_check.txt`). Simulated logs have exactly the registered shape: 5 arms × 8 supervisors × 105 items, with 20 banks and 25 rule items. They are drawn from the one-factor error model with planted per-arm precision τ and shared-error share r and analysed with `analyze_main.py` (Holm, one-sided, 5%). There were 100 replications per scenario, with B = 200 bootstrap draws inside each; the registered analysis uses 2,000.

| Scenario | H1 | H2a | H2b | H3 | H4 |
|---|---|---|---|---|---|
| Null: all arms τ = 1.0, r = 0.4 (rejection rate = size) | 0.01 | 0.02 | 0.04 | 0.01 | 0.00 |
| Planted effects: r = 0.8 / 0.5 / 0.6 / 0.3 (SS / SD / PS / PD); N0 τ = 0.2; novel items r + 0.15, τ − 0.3 | 1.00 | 0.85 | 0.66 | 0.40 | 0.77 |
| Pilot-like: r = 0.9 / 0.6 / 0.7 / 0.4, no novel-item effect | 1.00 | 0.88 | 0.71 | 0.29 | 0.01 |

How to read the table:
- Size is at or below 4% for every test.
- In the pilot-like scenario no H4 effect is planted, so its 0.01 is a size, not a power.
- With the first draft's 5 rule items, power under planted effects was 0.72 (H2a), 0.56 (H2b), 0.36 (H3) and 0.69 (H4). The 20 added rule items (Section 3.1) raise each of these.
- H3 remains underpowered (about 0.3–0.4), because the systemic-miss rate is a coarse, item-level count. H3 is still tested and reported as registered, but a non-rejection of H3 will be read as inconclusive, not as evidence against the hypothesis.
- The sample is limited by device time: about 35 s per call, so 4,200 calls take about 41 hours on one phone.

### 3.5 Stopping rule
The run stops when all 4,200 cells are complete. Progress is monitored only by counts; no contrasts are computed before completion.

## 4. Variables

### 4.1 Manipulated variables
- background corpus (shared / private)
- rule wording (same / different)
- rule passage (present / absent)

### 4.2 Measured variables
For each call:
- the answer;
- **miss** = 1 if the answer differs from the rule-based ground truth;
- whether the rule passage was retrieved;
- latency and the device clock cap (descriptive only).

### 4.3 Indices
Per arm and item class:
- miss rate p;
- precision τ = −Φ⁻¹(p);
- **shared-error share r**, the tetrachoric correlation of misses, obtained by solving Φ₂(−τ, −τ; r) = p₁₁, where p₁₁ is the pooled pairwise co-miss rate over the 28 supervisor pairs (r is set to 1 if p₁₁ exceeds the attainable maximum);
- **empirical systemic-miss rate**, the share of items on which more than 4 of 8 supervisors miss;
- model-implied P_sys at q = 0.5.

## 5. Analysis plan

### 5.1 Statistical models and tests
The estimates are as in Section 4.3, computed by `analyze_main.py`. For each hypothesis the contrast is:
- H1: miss(N0) − miss(PD), rule + covered items;
- H2a: [r(SS) + r(PS) − r(SD) − r(PD)] / 2, rule + covered items;
- H2b: [r(SS) + r(SD) − r(PS) − r(PD)] / 2, rule + covered items;
- H3: systemic-miss(SS) − systemic-miss(PD), rule + covered items;
- H4: mean over SS, SD, PS, PD of r(novel) − r(covered).

**Inference.** Cluster bootstrap: each bank's four items form one cluster, and each rule item is its own cluster (rule items are not tied to a bank), giving 45 clusters. There are 2,000 resamples with seed 7. The one-sided p-value is the share of bootstrap contrasts ≤ 0; 95% percentile intervals are reported.

### 5.2 Inference criteria
Holm correction across the five tests at a family-wise 5% level. A hypothesis is supported if its Holm-adjusted one-sided p < 0.05. Effects in the opposite direction and null results are reported in full.

### 5.3 Data exclusion
- banks identified in the masking check (Section 2.2);
- items with any missing cell (not expected; reported if it occurs).

No other exclusions. Answers cannot be unparseable because of the grammar constraint.

### 5.4 Exploratory analyses (not confirmatory)
- r and miss rate by wording;
- results conditional on retrieval of the rule passage;
- Fleiss' κ of misses;
- results by bank stratum, and for the three 2023 failures;
- per-question results;
- the relation between the device clock cap and latency.

## 6. Other

**Model and software:**
- Gemma 4 E2B instruction-tuned, Q4_0 GGUF (ggml-org/gemma-4-E2B-it-GGUF; SHA-256 8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52);
- llama.cpp commit 36e36e6, CPU-only, on a smartphone (8 cores); llama-server, context 4,096;
- Python 3 with NumPy and SciPy for the analysis.

**Limitations stated in advance:**
- one small model;
- simulated supervisors are not real supervisors;
- three wordings per rule, so private-wording arms are only partly disjoint;
- C1 has a single true category in this sample (all banks are well capitalised), so it measures false alarms only.

**Files and hashes at registration** (SHA-256; locked 24 Oct 2026; full list of 31 files in `SHA256SUMS_E1_registration.txt`, verifiable with `sha256sum -c`).

| File | SHA-256 |
|---|---|
| `e1_lib.py` | `68e1c2c15851c079762954afc35a1ecf92fcf80fb5826f2df7692664187f7012` |
| `run_main.py` | `d1a1e7795764f9c65166bfaf07a387d9deec2d3c74728c0c7abd19ae0f5e1b51` |
| `run_pilot3.py` | `7b8ffe540216634a47caba74869147261c1ee0676a4428d03ac18a4e1c0e739f` |
| `analyze_main.py` | `4d6a2ea48391b50f7338652d2ac730f7cf6b44d6b0280418b7b11d35bca10a77` |
| `sim_check.py` | `b755540bd7e7e4fcc799d252faa5fada9e0cbf524cb4d0d0b4cedd8e16515a26` |
| `power_check.py` | `3abd474de521833a2bb466cf944aaea196f281a1c2129f465bd897a76ac5330f` |
| `sample_banks.py` | `b28390cd2395770aa7170df3318ac7a70dd0ce42f0d95ad8cf9c91f758d8370d` |
| `build_gold.py` | `5151e41f5a57fd594877e01c5ab738b3a8061b65deaeed03481530bfffd17830` |
| `fetch_fdic.py` | `0c408861e02fd1edb704c98e5c291a405bdab484706f1c1760ab17cbdd708881` |
| `resume_e1.sh` | `e584613707688b8fbba8d16e749621c526abadd7d8708e1a0c2cb42976352c62` |
| `gold_passages.json` | `dfcbcc83eb8341c8a8d24bec1ba544146142e3d4ca423fcb03091feb370abab8` |
| `cases_20221231.csv` | `d0329fa1dfeddeb80a99eb74ddd9650570e4d8fbe0f1ba8372411dfcf62d213c` |
| `sample_banks.csv` | `cd1012319a15c98f3642c8fc516777ef3f89a3747f8c2ad3954cc6182494fda1` |
| `corpus_manifest.json` | `4841846cfc5097fc9dd1f1142c7c456369a600677f7667a732c82c7df30f6635` |
| `power_check.txt` | `b93b572998876a5afeaa737b9706e2502e71999f24e8adcc685a8cd7c4671146` |

Corpus sources and pilot files:

| File | SHA-256 |
|---|---|
| `corpus_raw/ecfr_252_5_categories.xml` | `06d6b170e29b789a2673d39fc1e320d51511e6b245be2d470e6f10eee3611198` |
| `corpus_raw/ecfr_324H_PCA.xml` | `7e3568c85f005246258dd0a75531f116fa0594484172d5fd3270b35b1380b993` |
| `corpus_raw/ecfr_324_10_minimums.xml` | `702646234e0dccd847ad0ba42ba4a58d5c9786514977e4226575f286e23406d0` |
| `corpus_raw/ecfr_324_22_adjustments.xml` | `38e805243c9944da3d1012c70b72f67c77895c875bd1bf96ea691ec4951a5a11` |
| `corpus_raw/ecfr_329_LCR_scope.xml` | `1ccac4e188e836376c07c2dc66250e2700d4eab6843b7e92e1c090b7da799fa5` |
| `corpus_raw/fdic_rms_2-1_wb20221204.pdf` | `9b4df4da795e95df81d974866e69a1468bcba77fef9200032912149c788b1ca3` |
| `corpus_raw/fdic_rms_6-1_wb20221210.pdf` | `1e2bf800e9bf591cd4bdf1d2f9fbfba85d86c40d58c4db0f37c52679e76298f5` |
| `corpus_raw/fed_cbem_wb20230120.pdf` | `9b0a654de6049371379300719b558988085e009599af7780d1c86a65c8373c34` |
| `corpus_raw/ffiec_031_041_instr_202212.pdf` | `3270c544bccad52ef1621003b67f39462efbcd50c0ffa3c417353ee90a02081d` |
| `corpus_raw/fr_84FR59230.txt` | `e5637fb6a344d3dc8c314141bccd42ceb9dd083628d1725195ffd4bb298651ba` |
| `corpus_raw/occ_ch_liquidity_wb20221016.pdf` | `688078698bfd739f01cbe68b9db7a6bcede1f571a800c71883c7a377c3f9298a` |
| `run_pilot.py` | `a97b95b88ef4b9228f35f4262c3cd8bd4a278cdcd176dd0a2800a980684b2488` |
| `run_pilot2.py` | `afa5767afd8738bef1b8ce1ad81bbf7cb21af2fbcb61c6ea15020a5421dab0ab` |
| `pilot_log.jsonl` | `ba7790f44ca008809b4b191ab1f63ad17629bf89c450b3857d9f640772767f4f` |
| `pilot2_log.jsonl` | `707c83019347058c756747bea7d274b2ce9e7df51d2afbebd7d471142737dace` |
| `pilot3_log.jsonl` | `b1526c1c8fb6c8a562272b3cd37ead0652080188b34140cb43d80b8f5818569c` |

Model file: `gemma-4-E2B-it-Q4_0.gguf`, SHA-256 8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52. llama.cpp commit 36e36e6.
