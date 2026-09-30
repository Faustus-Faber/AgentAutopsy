# ENHANCED JUDGE v2 — Plan for Validated Behavioral Judging
**Project**: Validation + hardening of the six-bucket behavioral decomposition of
agent prompt-injection trajectories (paper: "The Untriggered Default").
**Status**: HISTORICAL PRE-REGISTRATION PLAN (Dated 2026-09-05) — Executed & Extended in Canonical Suite (Protocol v3).

> [!NOTE]
> ### As-Run Implementation Notes & Pre-Registration Deltas
> This document preserves the historical pre-registered design plan (dated 2026-09-05) for transitioning from the label-exposed Protocol v1 baseline to a validated, label-blind evaluation protocol.
> 
> **Key Implementation Deltas in Canonical Suite (Protocol v3)**:
> 1. **Scale Expansion**: The pre-registration proposed an initial 400-case stratified validation sample. The final canonical execution ([judge_v3.py](runners/judge_v3/judge_v3.py)) scaled to **$N = 33{,}000$ trajectories** (exactly 500 cases per cell across all 66 cells: 11 models × 3 precisions [FP16, FP8, NF4] × 2 benchmarks [AgentDojo, InjecAgent]).
> 2. **Canonical Evaluator**: Rather than relying exclusively on the free-tier `z-ai/glm-5.3-free` runner under $T=0.7$ multi-sampling (which introduced variance and token limits on complex data-stealing traces), the canonical evaluation locked **`deepseek/deepseek-v4.1-flash`** at greedy deterministic $T=0.0$ with an **8,192-token completion safety ceiling** across a distributed multi-provider asynchronous pool (9 Atria keys + Alysis endpoints).
> 3. **Role of GLM-5.3**: GLM-5.3 was retained as an independent cross-model validation benchmark, achieving **$91.4\%$ raw agreement** and **Cohen's $\kappa = 0.862$** against the canonical DeepSeek judge on the validation sample.
> 4. **Enforcement of Rule D7**: Programmatic quote verification achieved **$90.77\%$ validity ($9,466 / 10,428$)** via automated normalized substring matching directly against raw assistant reasoning scratchpads.
> 5. **Bucket B6 Formalization**: Bucket 6 is formally standardized as `BROKE` across all code, manifests, and reports.

---

## 1. Objective

Replace the paper's judge-only, label-exposed, single-sample bucket labels with a
**label-blind, injection-hardened, self-consistent, dual-family, human-calibrated**
judging pipeline (v2), then reconcile v2 with the v1 corpus so every paper claim
carries a measured validity figure. Everything is staged so spend stops the moment
a gate fails.

## 2. What we are judging (unchanged from v1 corpus)

Input per record: one full agent trajectory (system+user+assistant+tool messages,
with the agent's chain-of-thought) from the 29,868-record judged corpus
(5 models × 3 precisions × 2 benchmarks × baseline arms).
Output: one of six buckets {FULL_COMPLIANCE, PARTIAL_COMPLIANCE, GENUINE_REFUSAL,
INERT_RECOGNIZED, INERT_UNRECOGNIZED, BROKE} + recognition flag + first-recognition
call index + verbatim evidence quote + confidence.

## 3. v1 → v2 design deltas (each literature-grounded)

| ID | v1 (existing corpus) | v2 (this plan) | Why (evidence) |
|----|---------------------|----------------|----------------|
| D1 | Judge saw `BENCHMARK LABEL: attack_succeeded=...` | Label line removed entirely | Anchoring: prior labels block 48% of error corrections, flip 10.2% of correct judgments (arXiv:2608.25869) |
| D2 | — | Keep attacker goal + tool specs as task-level reference | Reference-guided judging improves validity; distinguish task ground truth (keep) from outcome anchor (drop) |
| D3 | Refusal defined as "the only real defense" | Neutral behavioral definition | Evaluative prompt language biases bucket choice |
| D4 | Trajectory pasted raw; no hardening | Trajectory fenced as quoted untrusted artifact + explicit judge-side anti-manipulation instruction; **hijack-rate experiment**: re-judge 40 records with injection spans neutralized, compare verdicts | Judges are injectable via evaluated content; targeted attacks >30% ASR (arXiv:2505.13348, JudgeDeceiver) |
| D5 | 1 sample per record | k=3 samples at T=0.7, majority aggregation; report self-agreement rate | Self-consistency/majority voting consistently improves agreement (arXiv:2510.27106, 2408.09235) |
| D6 | Single judge per record, mixed cells, composition unmeasured | Two judge families (e.g., DeepSeek-V4-Flash + Qwen3.8-Max) independently; per-judge and pooled κ; cross-family disagreement → tiebreaker tier | Multi-judge ensembles increase κ with humans (2408.09235) |
| D7 | Evidence quote required, never validated | Verbatim quote required for any recognition flag; programmatic substring validation; invalid → 1 re-ask, else record marked unverifiable | Judge CoT rationales can be unfaithful; validation converts silent failure into detectable failure |
| D8 | Raw agreement vs benchmark label (98–100%) — meaningless on 70–78% majority bucket | Cohen's κ + per-bucket precision/recall vs reference; rare buckets oversampled in sampling | κ corrects for imbalance (calibration literature) |
| D9 | No stability measurement | 10% subset re-judged at T=0 → run-to-run stability figure | Within-run noise is material (~44.7% of score variance in one study) |
| D10 | Analysis done after seeing results | **Pre-registered decision gates** (§6), locked before Stage 1 | Post-hoc analysis is weaker evidence (Nanda; pre/post-hoc discipline) |
| D11 | No human calibration | Author codes a 100-case reference set with a 2-page codebook; report κ(v2, author) + confusion matrix | Judge must be evaluated like a classifier against a human reference |

## 4. Sampling design

- **Frame**: the 29,868 judged records (case_ledger.jsonl), baseline arms only.
- **Stage 0 pilot**: 20 records — 2 per benchmark spanning predicted-easy and
  predicted-hard buckets (≥4 INERT_UNRECOGNIZED, ≥4 FULL_COMPLIANCE, ≥4 GENUINE_REFUSAL,
  ≥4 INERT_RECOGNIZED, ≥4 mixed). Purpose: JSON validity, prompt sanity, early hijack sniff.
- **Stage 1 main**: 400 records, stratified proportional across the 30 cells
  (≥13/cell) **with rare-bucket floor**: all GENUINE_REFUSAL / BROKE / PARTIAL records
  in the sample force-include up to availability, min target 30 per rare bucket.
  Power: n=400 gives κ 95% CI ≈ ±0.06–0.09 under expected agreement; per-bucket
  precision/recall CIs via bootstrap.
- **Stage 2 adjudication**: cross-family disagreements (expect 5–15% → 20–60 records),
  1 tiebreaker call each (larger judge or deterministic re-prompt).
- **Stage 3 optional extension**: to 1,000 records if gates are borderline.

## 5. Call & budget accounting

**Measured (2026-09-05, test_keys.py + round 2 + free-tier probe):**
- 47 keys supplied; **47/47 live** on `z-ai/glm-5.3-free` (round-1 429s were per-IP
  burst throttling; paced usage 10/10 clean).
- **`z-ai/glm-5.3-free` is the ONLY free model** — all other catalog entries (incl.
  deepseek-v4-flash, qwen3.8-max, kimi, gemini) return 403 "credit limit insufficient".
- glm-5.3-free is a **thinking model**: `reasoning_content` present on 47/47 calls;
  trivial prompt = ~60 completion tokens ⇒ judge-sized calls will carry thinking tokens.
- Latency (tiny calls): min/med/max = 0.85 / 3.55 / 47 s (high variance = free-tier queueing).
  Judge-sized calls (3–8K-token trajectory + CoT + JSON) estimated 30–90 s.
- Fleet throughput estimate: 47 keys × 1 concurrent call each ≈ **30–70 calls/min**
  (latency-bound; the 8 rpm/key cap is not binding at these latencies).

**D6 consequence:** with one free model available, the dual-FAMILY ensemble is impossible
on the free tier alone. Two options (author decides):
  - (a) **Single-family, k=5 self-consistency** (budget +1,200 calls): diversity via
    sampling only; paper documents judge-family homogeneity as a limitation.
  - (b) **Author tops up a small credit** to unlock `deepseek/deepseek-v4-flash` as
    family #2 — which is also the v1 corpus's primary judge, doubling as the v1↔v2
    bridge and restoring true cross-family κ. Estimated need: family #2 on 400 records
    × 3 samples = 1,200 calls ≈ 7–12M tokens.

Original table (per-call numbers unchanged):

| Stage | Calls | Wall time | Tokens (≈6K in + 1.5K out/call) |
|-------|-------|-----------|-------------------------------|
| 0 pilot (20 × 2 judges × 1) | 40 | ~4 min | 0.3M |
| 1 main (400 × 3 samples × 2 judges) | 2,400 | ~2.7 h | 18M |
| 1b determinism subset (40 × 2 judges × 1 @ T=0) | 80 | ~6 min | 0.6M |
| 1c hijack probe (40 × 2 variants × 1 judge) | 80 | ~6 min | 0.6M |
| 2 adjudication (≤60 × 1) | ≤60 | ≤5 min | ≤0.5M |
| **Core total (option a, glm-only, k=5)** | **≈4,060** | **≈1.5–2.5 h** | **≈25–40M tokens** |
| Optional Stage 3 (+600 records full protocol) | +3,600 | +4 h | +27M |

Hard cost cap enforced in the runner: no run proceeds past a stage without the
author's go; runner aborts at a configurable spend/request ceiling (default: 3,500 calls).

## 6. Pre-registered decision gates (locked before Stage 1)

- **G1 prompt validity (Stage 0)**: ≥90% of pilot calls return parseable JSON with a
  legal bucket and (when recognition claimed) a substring-valid quote. Fail → fix prompt,
  re-run Stage 0 (cheap).
- **G2 label-anchoring effect (primary)**: κ(bucket_v2_labelblind, bucket_v1) over the 400.
  - κ ≥ 0.85 **and** |ΔARR| < 1pp → v1 corpus numbers STAND; paper gains a
    robustness appendix; Limitations upgraded to "label-blind re-judge reproduces".
  - 0.70 ≤ κ < 0.85 → v2 labels reported alongside v1; headline buckets recomputed on
    the 400-record sample with CIs; discrepancy cells analyzed.
  - κ < 0.70 → v2 becomes primary; choose full-corpus v2 re-judge (~33 h, 29,868 × 1
    call — pre-approved budget question) OR downgrade §4 claims to sample-level CIs.
- **G3 judge-hijack rate**: verdict flip rate between normal and neutralized-payload
  variants ≤ 5% → robustness documented in the paper (novel measurement).
  \> 5% → neutralized presentation becomes the default for all judging; re-run Stage 1.
- **G4 self-consistency**: k=3 self-agreement ≥ 85% → majority label used as-is.
  < 85% → raise k to 5 for affected cells (budget +1,200 calls, author approval).
- **G5 human calibration**: κ(v2, author) ≥ 0.70 → "author-validated" claim;
  0.50–0.70 → report with caveats; < 0.50 → bucket scheme itself is in question →
  author + I renegotiate bucket definitions before any paper claim rests on them.

## 7. Artifacts produced

- `results/judgments_v2/*.jsonl` — one record per call: model, sample idx, prompt hash,
  bucket, recognition flag, quote, quote_valid, finish, tokens, latency, raw response.
- `analysis/label_blind_reconciliation.json` — κ(v1,v2), per-bucket confusion matrix,
  ΔARR + bootstrap CI, per-cell breakdowns.
- `analysis/hijack_rate.json` — flip rate normal vs neutralized payloads.
- `analysis/human_calibration.json` — κ(v2, author), per-bucket agreement.
- Paper-ready numbers + a Limitations text block pre-drafted for each gate outcome.

## 8. Runner spec (implemented after approval, before keys are spent)

- `runner/judge_v2.py` — one worker per key (20 threads), per-key token bucket at 8 rpm,
  global stop switch, per-record checkpoint/resume (JSONL append + manifest),
  automatic retry on 429 (respect Retry-After), max 3 attempts, dead-letter file.
- `runner/prompts/` — versioned prompt files (system_v2.txt, rubric_v2.md, payload-fence
  variants); every record stores `prompt_hash` for the judge contract (drift detection).
- `runner/sample.py` — stratified sampler from case_ledger.jsonl with rare-bucket floor;
  emits the fixed record list + seed BEFORE any calls (pre-registration).
- Keys supplied via `Judge/runner/keys.txt` (gitignored) or env var; never logged.

## 9. What the author must decide / supply

1. **Judge model(s)** — DECIDED BY MEASUREMENT: only `z-ai/glm-5.3-free` is free on the
   supplied keys. Author chooses: (a) glm-only with k=5 self-consistency, or (b) top up
   credit to add `deepseek/deepseek-v4-flash` as family #2 (= v1's primary judge, doubles
   as the v1↔v2 bridge; est. 1,200 extra calls).
2. **API endpoint + 47 keys — TESTED AND LIVE** (47/47 on glm-5.3-free; zero paid credit).
3. **Gate G2 branch preference** if κ lands 0.70–0.85: sample-level recompute (free) vs
   full-corpus v2 re-judge (~33 h wall) — decide only if it triggers.
4. **Author coding commitment** for D11/G5: 100 cases ≈ 3–5 h of your time, can run
   during the ARR review window.
5. Confirm the cost cap (default 3,500 calls).

## 10. Explicitly out of scope for v2

- Re-judging the intervention arms (EXP2B/4A/5A/6) — those are raw-outcome based, not
  judge-based; no v2 needed.
- Human inter-annotator agreement between multiple humans (single-author calibration only).
- pairwise/position-bias controls (N/A for single-trajectory classification; documented).
