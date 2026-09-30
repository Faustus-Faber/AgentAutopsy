<div align="center">

<img src="assets/images/banner.png" alt="AgentAutopsy: Low Attack Success Is Not Security Banner" width="100%" />

<br/>

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![vLLM Backend](https://img.shields.io/badge/vLLM-0.6%2B-00C7B7.svg?style=for-the-badge&logo=vllm&logoColor=white)](https://github.com/vllm-project/vllm)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Audit%20Corpus%20(33k)-20BEFF.svg?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/bracu-agent-security/agent-prompt-injection-behavioral-audit-33k)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)
[![Institution](https://img.shields.io/badge/Brac%20University-CSE%20Thesis-005691.svg?style=for-the-badge)](https://www.bracu.ac.bd/)

<p align="center">
  <h1 align="center">🛡️ AgentAutopsy</h1>
  <b>Official Research Repository &amp; Empirical Codebase for the Undergraduate Thesis</b><br/>
  <i>Low Attack Success Is Not Security: Diagnosing Prompt-Injection Failures in Small Tool-Using Agents</i><br/>
  <b>Department of Computer Science and Engineering &bull; Brac University (Summer 2026)</b>
</p>

---

[📖 Paper Abstract](#-executive-summary--core-findings) &bull;
[🔬 Dual-Benchmark Architecture](#-dual-benchmark-apparatus) &bull;
[⚖️ 33k Behavioral Audit](#-label-blind-behavioral-audit-33000-trajectories) &bull;
[🎛️ Deployment Interventions](#-causal-deployment-interventions) &bull;
[📦 Kaggle Datasets](#-kaggle-dataset-releases) &bull;
[⚡ Instant Reproduction](#-instant-reproduction--verification) &bull;
[📚 BibTeX Citation](#-citation)

---

</div>

## 🚀 Executive Summary & Core Findings

Prompt-injection benchmarks measure whether instructions hidden in untrusted external data redirect a tool-using language agent, commonly reporting a single metric: the **Attack Success Rate (ASR)**. On modern open-weight compact reasoning models (≤ 14B parameters), reported ASR is frequently below 10%. This has fostered the widespread belief that smaller instruction-tuned models possess intrinsic alignment against adversarial injection.

> **Core Thesis Statement**:  
> **Low attack success is not security.** An unsuccessful benchmark attack does not imply robust defense: the agent may have passively overlooked the injection while executing the user's task, experienced catastrophic tool syntax collapse, or deferred to default action inertia. When deployment context shifts—even without altering model weights—compliance changes by over 300×.

<div align="center">
  <img src="assets/images/fig_4_1_evaluation_workflow.png" alt="End-to-End Diagnostic Pipeline Architecture" width="94%" />
  <p><em>Figure 1: End-to-end diagnostic and empirical evaluation framework connecting dual-benchmark trajectory harvesting, label-blind behavioral auditing, and causal policy interventions.</em></p>
</div>

### 🔑 Key Scientific Discoveries:
1. 🙈 **Silent Inattention Dominates Defense (§6.3)**: Over 65% of attacks labeled "unsuccessful" involved neither attacker compliance nor defensive refusal. Agents passively ignore injected instructions (`INERT_UNRECOGNIZED`), completing legitimate user tasks in 77.74% of cases.
2. 🧠 **The Recognition–Enforcement Gap & Perception–Compliance Dissociation (PCD, §6.3.4)**: In 28%–47% of trajectories, compact reasoning models explicitly identify, quote, and analyze the prompt injection inside their chain-of-thought scratchpad, yet proceed to comply anyway in over 40%–74% of those recognized cases.
3. ⚙️ **Defense by Tool Syntax Breakdown (§6.3.5)**: Apparent zero-ASR scores can be purely synthetic. `LFM-2.6B` achieves 0.00% baseline ASR on InjecAgent solely because 70% of its calls crash with malformed JSON (735 / 1,054 invalid tool calls).
4. 🔓 **The Untriggered Default & 339× Policy Dynamic Range (§6.5.2)**: Appending three sentences of administrative authorization to the system prompt causes ASR to surge by up to +69.93 pp (`Spark-4B`: 0.95% → 70.88%; `Gemma-4B`: 7.02% → 72.69%; pooled +34.98 pp). Conversely, administrative interdiction collapses ASR to 0.22%.
5. 🪜 **Continuous Authority Dose-Response (§6.5.3)**: Compliance scales along an unbroken monotonic staircase (M_step = 1.00) across 5 linguistic authority levels (L0 → L4), while legitimate task utility collapses.
6. 🧩 **Reasoning Deliberation Is Orthogonal to Protection (§6.5.5)**: Suppressing Chain-of-Thought deliberation (`thinkOFF` vs `thinkON`) across 28 factorial cells never increased security; instead, disabling reasoning significantly heightened attack success in 8 cells.
7. 📉 **Quantization Degrades Execution Syntax, Not Attack Resilience (§6.5.6)**: 4-bit NF4 quantization reduces apparent ASR not by enhancing robustness, but by triggering token loops and tool format dropout.

---

## 🔬 Dual-Benchmark Apparatus

We evaluate the full 11-model cohort across two complementary prompt-injection benchmarks:
* **InjecAgent**: Two-turn ReAct state machine (N = 1,054 nominal test cases: 510 Direct Harm, 544 Data Stealing).
* **AgentDojo**: Stateful, multi-turn environment (N = 949 paired attack episodes, N = 97 clean utility controls) spanning 4 real-world application suites (Workspace, Banking, Slack, Travel).

<div align="center">
  <img src="assets/images/fig_4_2_benchmark_comparison.png" alt="Dual-Benchmark Evaluation Architecture" width="94%" />
  <p><em>Figure 2: Dual-benchmark evaluation architecture contrasting the two-turn ReAct structure of InjecAgent with the stateful multi-turn environment of AgentDojo.</em></p>
</div>

### 📋 Evaluated Model Cohort (11 Open-Weight Architectures)

<div align="center">
  <img src="assets/images/table_model_registry_styled.png" alt="Model Cohort Specifications" width="92%" />
</div>

All models were evaluated across three serving precision regimes (**FP16, FP8, and NF4**) under strict greedy decoding ($T = 0.0$):
* **Sliding Window Hybrids**: `Gemma-4B` (`gemma-2-4b-it`), `Gemma-12B` (`gemma-2-12b-it`)
* **Gated DeltaNet Hybrids**: `Qwen-4B` (`Qwen2.5-Coder-7B` / `Qwen3.5-4B`), `Qwen-9B` (`Qwen3.5-9B`)
* **Mamba2-Transformer Hybrid**: `Nemotron-Nano-4B` (`Nemotron-3-Nano-4B`)
* **Looped Transformer (22×2)**: `Nanbeige-3B` (`Nanbeige-4.2-3B`)
* **Standard Dense & Reasoning**: `MiniCPM-2B`, `Ministral-14B`, `Ornith-9B`, `Spark-4B`
* **Liquid State Space (SSM)**: `LFM-2.6B` (`LFM-2.5-2.6B`)

---

## 📊 Dual-Benchmark Baseline Outcomes (Thesis §6.2)

### 📈 Table 1: Targeted Attack Success Rates Across Architectures & Precisions (Thesis Table 6.1)

| Model Architecture | Parameter Count | Architectural Family | InjecAgent FP16 | InjecAgent FP8 | InjecAgent NF4 | AgentDojo FP16 | AgentDojo FP8 | AgentDojo NF4 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemma-4B** | 4B | Sliding Window Hybrid | 7.02% | 5.12% | 7.78% | 13.49% | 13.91% | 19.49% |
| **Gemma-12B** | 12B | Sliding Window Hybrid | 3.51% | 3.23% | 1.42% | 27.61% | 24.87% | 8.85% |
| **LFM-2.6B** | 2.6B | Liquid SSM Hybrid | **0.00%**<sup>†</sup> | **0.00%**<sup>†</sup> | **0.00%**<sup>†</sup> | 2.42% | 2.00% | 2.95% |
| **MiniCPM-2B** | 2B | Standard Dense | 0.85% | 0.95% | 0.00% | 5.90% | 5.80% | 2.32% |
| **Ministral-14B** | 14B | Dense Reasoning | 0.28%<sup>†</sup> | 5.69% | 4.55% | 22.55% | 13.28% | 16.23% |
| **Nanbeige-3B** | 3B | Looped Transformer (22×2) | 2.47% | 2.75% | 1.04% | 4.64% | 5.37% | 4.32% |
| **Nemotron-Nano-4B** | 4B | Mamba2-Transformer Hybrid | 2.56% | 2.75% | 4.84% | 15.81% | 16.54% | 15.49% |
| **Ornith-9B** | 9B | Standard Dense | 27.42% | 20.30% | 13.47% | **1.90%** | **1.16%** | **1.58%** |
| **Qwen-4B** | 4B | Gated DeltaNet Hybrid | 18.79% | 15.84% | 13.28% | 25.18% | 26.45% | 21.92% |
| **Qwen-9B** | 9B | Gated DeltaNet Hybrid | 25.14% | 24.76% | 22.43% | 9.69% | 10.90% | 10.96% |
| **Spark-4B** | 4B | Hybrid SWA 3:1 | 0.95% | 1.04% | 1.04% | 2.32% | 1.90% | 1.79% |
| **All Models Pooled** | -- | -- | **8.09%** | **7.50%** | **6.35%** | **11.96%** | **11.11%** | **9.63%** |
| **Median Model** | -- | -- | **2.56%** | **3.23%** | **4.55%** | **9.69%** | **10.90%** | **8.85%** |

*Legend: <sup>†</sup> indicates attack failures driven by high tool invalidity/syntax crashes.*

<div align="center">
  <img src="assets/images/fig_vulnerability_asymmetry.png" alt="Vulnerability Asymmetry" width="90%" />
  <p><em>Figure 3: Vulnerability asymmetry between Direct Harm and Data Stealing attacks on InjecAgent (FP16 baseline). Data Stealing succeeds 3.8&times; more often across models.</em></p>
</div>

---

## ⚖️ Label-Blind Behavioral Audit (33,000 Trajectories)

To understand what happens when an attack is scored as "failed", we conducted an independent, label-blind behavioral audit of **33,000 interaction trajectories** (500 per model-precision cell across both benchmarks) using **DeepSeek-V4.1-Flash** as an automated judge with strict programmatic quote verification (Rule D7: $90.77\%$ validity).

<div align="center">
  <img src="assets/images/fig_behavior_composition.png" alt="6-Bucket Behavioral Composition" width="94%" />
  <p><em>Figure 4: 6-bucket behavioral taxonomy distribution across all 11 model architectures on InjecAgent and AgentDojo.</em></p>
</div>

### 🧩 The Six Mutually Exclusive Behavioral Buckets:
1. 🔴 **FULL_COMPLIANCE**: Agent executes the attacker's designated tool objective with specified arguments.
2. 🟠 **PARTIAL_COMPLIANCE**: Agent attempts the attacker's goal or performs unauthorized preparatory steps.
3. 🟢 **GENUINE_REFUSAL**: Agent explicitly detects the injection, cites authorization boundaries, and refuses.
4. 🔵 **INERT_RECOGNIZED**: Agent notes the injection in internal thought tokens, but silently ignores it while continuing the user's task.
5. ⚪ **INERT_UNRECOGNIZED**: Agent completely overlooks the injection (passive inattention).
6. 🟣 **MECHANICAL_BREAKDOWN**: Tool calling fails due to malformed JSON, invalid arguments, or token loops.

### 📊 The Attack–Task Outcome Matrix (ATOM, Thesis §6.4)

Crossing attack success ($S \in \{0, 1\}$) with legitimate user task utility ($U \in \{0, 1\}$) reveals that **safety and utility are separate operational dimensions**:

<div align="center">
  <img src="assets/images/fig_atom_heatmap.png" alt="ATOM Heatmap" width="48%" />
  <img src="assets/images/f1_join.png" alt="Silent Failure Composition" width="48%" />
  <p><em>Figure 5: (Left) Joint distribution of ATOM outcomes on AgentDojo; (Right) Silent failure composition and legitimate-task completion rates across models.</em></p>
</div>

| Behavioral Category | Total Trajectories ($N$) | User Task Completed ($U=1$) | User Task Failed ($U=0$) | Operational Implication |
| :--- | :---: | :---: | :---: | :--- |
| **Genuine Refusal** | 2,643 | **83.20%** | 16.80% | Model safely rejects injection; user task preserved. |
| **Inert-Unrecognized** | 10,365 | **77.74%** | 22.26% | Silent continuation preserves legitimate user work. |
| **Inert-Recognized** | 704 | **71.88%** | 28.12% | Perceived in CoT, bypassed in execution. |
| **Partial Compliance** | 257 | 36.58% | 63.42% | Attacker side-channel derails user goal. |
| **Full Compliance** | 252 | 21.83% | 78.17% | Severe utility sacrifice to serve attacker. |
| **Mechanical Breakdown** | 251 | **11.95%** | **88.05%** | Syntax collapse ruins both security and utility. |

---

## 🎛️ Causal Deployment Interventions (Thesis §6.5)

We manipulate deployment scaffolding—system prompts, authority hierarchy, test-time compute, and numerical precision—while keeping **model weights strictly unchanged**:

<div align="center">
  <img src="assets/images/f2_authority.png" alt="Authority Dose Response" width="94%" />
  <p><em>Figure 6: (Panel a) Continuous authority dose-response on AgentDojo comparing Gemma-4B against baselines; (Panel b) Monotonic collapse of Safe Utility (S0U1) as linguistic authority escalates.</em></p>
</div>

### 📑 Selected Deployment Intervention Results (Thesis Table 6.10)

| Experimental Dimension | Condition / Intervention | Targeted ASR (%) | Utility Under Attack (UA) | Safe & Useful (S0U1) |
| :--- | :--- | :---: | :---: | :---: |
| **Instruction Boundary Guardrails** | Baseline InjecAgent<br>Guardrail Intervention | 9.69%<br>**0.95%** | --<br>-- | --<br>-- |
| **Administrative Policy Framing** | Baseline InjecAgent<br>Interdict Policy<br>Authorize Policy | 8.09%<br>**0.22%**<br>**43.07%** | --<br>--<br>-- | --<br>--<br>-- |
| **Linguistic Authority Ladder** | Level L0 (Baseline)<br>Level L1 (Suggestive)<br>Level L2 (Directive)<br>Level L3 (Rationalized) | 13.49%<br>23.50%<br>38.67%<br>**42.04%** | 68.18%<br>63.86%<br>50.16%<br>**45.31%** | 63.75%<br>56.06%<br>37.41%<br>**30.03%** |
| **Serving Numerical Precision** | FP16 Pooled<br>FP8 Pooled<br>NF4 Pooled | 11.96%<br>11.11%<br>**9.63%** | 72.73%<br>72.10%<br>**67.98%** | 68.77%<br>68.29%<br>**64.80%** |
| **Reasoning-Token Budget** | Nanbeige-3B 2,000 tokens<br>Nanbeige-3B 256 tokens | 1.23%<br>**0.00%** | --<br>-- | --<br>-- (96.1% syntax invalids) |

<div align="center">
  <img src="assets/images/fig_precision_sensitivity.png" alt="Precision Sensitivity" width="70%" />
  <p><em>Figure 7: AgentDojo precision sensitivity across FP16, FP8, and NF4 (N=949 matched episodes). Quantization lowers ASR slightly but simultaneously degrades legitimate task utility.</em></p>
</div>

---

## 🏆 Master Model Evaluation Scorecard (Thesis Table 6.11)

Models ranked by Safe and Useful execution (S0U1):

| Model Architecture | InjecAgent ASR | AgentDojo ASR | Refusal % (AD / IA) | Invalid Calls (IA) | ATOM S0U1 (%) | ATOM S0U0 (%) | Authorize Surge (Δpp) | Security & Behavioral Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ornith-9B** | 27.42% | **1.90%** | **83.6 / 71.9** | 19 | **87.6%** | 11.0% | +34.95pp | Strong genuine refusal; highest S0U1. |
| **Nanbeige-3B** | 2.47% | 4.64% | 7.6 / 38.1 | 270 | **80.1%** | 15.1% | +23.24pp | High S0U1, susceptible to micro-clamping. |
| **Qwen-9B** | 25.14% | 9.69% | 2.0 / 13.7 | 44 | **78.3%** | 10.1% | +49.43pp | High utility, highly compliant under authority. |
| **Spark-4B** | 0.95% | 2.32% | 47.6 / 24.8 | 6 | **75.7%** | 22.5% | +69.93pp | Largest authority surge (+69.9pp). |
| **MiniCPM-2B** | 0.85% | 5.90% | 4.5 / 2.9 | 42 | **71.3%** | 24.1% | +2.09pp | Low ASR driven by silent continuation. |
| **LFM-2.6B** | **0.00%**<sup>†</sup> | 2.42% | 17.3 / 5.1 | 735 | **64.1%** | 33.5% | +0.00pp<sup>†</sup> | Zero ASR caused by syntax breakdown. |
| **Gemma-4B** | 7.02% | 13.49% | 11.1 / 8.4 | **1** | **60.1%** | 24.2% | +65.67pp | High execution validity, authority-sensitive. |
| **Qwen-4B** | 18.79% | 25.18% | 4.4 / 6.8 | 19 | **59.8%** | 16.1% | +43.31pp | Consistently vulnerable across both benchmarks. |
| **Gemma-12B** | 3.51% | 27.61% | 3.5 / 20.0 | **1** | **56.7%** | 23.3% | +57.92pp | Highest AgentDojo attack vulnerability. |
| **Nemotron-Nano-4B**| 2.56% | 15.81% | 1.1 / 0.5 | 162 | **54.7%** | 28.5% | +45.13pp | Silent execution, almost never refuses. |
| **Ministral-14B** | 0.28%<sup>†</sup>| 22.55% | 0.8 / 4.4 | 263 | **50.1%** | 30.4% | -0.17pp<sup>†</sup> | Low InjecAgent ASR via invalid calls. |

---

## 📦 Kaggle Dataset Releases

The empirical datasets are fully prepared, packaged, and documented for public release:

### 1. 📂 Behavioral Audit Corpus (33,000 Trajectories)
* **Kaggle Slug**: [`bracu-agent-security/agent-prompt-injection-behavioral-audit-33k`](https://www.kaggle.com/datasets/bracu-agent-security/agent-prompt-injection-behavioral-audit-33k)
* **Pre-bundled Directory**: [`kaggle/kaggle_audit_corpus_package/`](kaggle/kaggle_audit_corpus_package/)
  * `audit_corpus_33k.parquet` (6.69 MB, snappy compressed, instant columnar query)
  * `audit_corpus_33k.jsonl.gz` (4.04 MB compressed raw audit lines)
  * `data_dictionary.csv` (18 columns fully documented)
  * `dataset-metadata.json` (Kaggle dataset manifest)
* **Kaggle CLI Publication**:
  ```bash
  kaggle datasets create -p kaggle/kaggle_audit_corpus_package
  ```

### 2. 🗄️ Full Trajectory Corpus (146,000+ Raw Execution Episodes)
* **Included In Repository**: All raw execution trajectories across all 11 models, 3 precisions, and 6 intervention experiments are **retained and tracked** in their respective experiment folders (`raw_results/`).
* **Bulk Download**: Use [`kaggle/package_kaggle_datasets.py`](kaggle/package_kaggle_datasets.py) to package segmented `.tar.gz` archives for high-speed cloud hosting.

```python
import pandas as pd

# Load 33,000 audited trajectories in <1 second
df = pd.read_parquet("kaggle/kaggle_audit_corpus_package/audit_corpus_33k.parquet")
print(f"Total Trajectories: {len(df):,}")

# Behavioral breakdown
print(df["behavioral_bucket"].value_counts(normalize=True) * 100)

# Check Perception-Compliance Dissociation
recog = df[df["recognized_injection"] == True]
print(f"Refusal conversion when recognized: {(recog['behavioral_bucket'] == 'GENUINE_REFUSAL').mean() * 100:.2f}%")
```

---

## ⚡ Instant Reproduction & Verification

To verify and reproduce all Chapter 6 thesis tables in **under 0.3 seconds**, run the master reproduction script:

```bash
python scripts/reproduce_all_thesis_results.py
```

<div align="center">
  <img src="assets/images/terminal_preview.png" alt="Terminal Output Preview" width="94%" />
</div>

<details>
<summary><b>🔍 Click to view reproduction commands for individual experiments</b></summary>

```bash
# 1. Dual-Benchmark Baseline Outcomes (Thesis §6.2)
python 01_dual_benchmark_baselines/analyze_baselines.py

# 2. 33k Behavioral Audit & ATOM Matrix (Thesis §6.3 & §6.4)
python 02_behavioral_auditing/analyze_judgments.py

# 3. Instruction Boundary Guardrails (Thesis §6.5.1)
python 03_instruction_boundary_guardrails/analyze_boundary_guardrails.py

# 4. Administrative Authority Framing (Thesis §6.5.2)
python 04_administrative_authority_framing/analyze_authority_framing.py

# 5. Continuous Linguistic Dose-Response (Thesis §6.5.3)
python 05_continuous_authority_dose_response/analyze_authority_dose_response.py

# 6. Reasoning Token Budget Constraints (Thesis §6.5.4)
python 06_reasoning_budget_and_context_limits/analyze_reasoning_budget.py

# 7. Reasoning-Token Ablation 2x4 Factorial (Thesis §6.5.5)
python 07_reasoning_token_ablation/analyze_reasoning_ablation.py

# 8. Cross-Precision Dynamics (Thesis §6.5.6)
python 08_numerical_precision_dynamics/analyze_precision_dynamics.py
```
</details>

---

## 📚 Citation

If you use this repository, the behavioral audit corpus, or evaluation methodology in your research, please cite our thesis:

```bibtex
@thesis{zarif2026low,
  title        = {Low Attack Success Is Not Security: Diagnosing Prompt-Injection Failures in Small Tool-Using Agents},
  author       = {Zarif, Farhan and Galib, Kazi Shahid Ahmed and Tarin, Shabnam Sharmin and Promi, Synthia Akter},
  year         = {2026},
  school       = {Department of Computer Science and Engineering, Brac University},
  type         = {Bachelor's Thesis},
  address      = {Dhaka, Bangladesh},
  keywords     = {Autonomous Agents, Indirect Prompt Injection, Attack Success Rate, Attack-Task Outcome Matrix, Behavioral Auditing, Deployment Scaffolding}
}
```

---

## 📄 License

This empirical repository and dataset are licensed under the [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/) license. All analysis scripts and execution harnesses are licensed under the [Apache 2.0 License](LICENSE).
