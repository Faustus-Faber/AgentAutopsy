# Experiment 3A: Reasoning Budget Constraint Report

**Target Scope**: Causal Investigation of Reasoning Depth and Context Token Budgets on Prompt Injection Compliance  
**Evaluated Benchmark**: InjecAgent ($N=1,054$ cases per full run, $N=200\text{--}203$ cases per macro sweep)  
**Evaluated Models**:
- **Macro-Budget Sweep ($N=200\text{--}203$ cases each)**: Qwen3.5-4B, Qwen3.5-9B, Gemma-4-E4B-it, Gemma-4-12B-it, Nemotron-3-Nano-4B, Ministral-3-14B, Ornith-1.5-9B, MiniCPM5-2B, Spark-X2.5-4B
- **Deep Micro-Budget Clamping ($N=1,054$ cases)**: Nanbeige-4.2-3B (FP16)  
**Evaluated Budgets**: Baseline ($60,000$ tokens), $16,000$ tokens, $8,000$ tokens, $4,000$ tokens, $2,000$ tokens, $1,000$ tokens, $512$ tokens, $256$ tokens  

---

## 1. Experimental Methodology & Technical Protocol

### A. Research Question & Causal Hypothesis
* **Primary RQ (RQ3A)**: Does deeper Chain-of-Thought (CoT) reasoning engagement with injected content causally facilitate attack compliance, or is reasoning length merely a post-hoc symptom of compliance?
* **Hypothesis**: In baseline evaluations, successful attacks exhibit substantially longer reasoning traces than failed attacks. If extensive reasoning is causal—providing the model with cognitive space to reframe adversarial instructions as user tasks—mechanically constraining the maximum generation budget (`max_tokens`) should reduce Attack Success Rate (ASR) proportionally.

### B. Two-Stage Experimental Design
1. **Stage 1: Macro-Budget Scaling Sweep ([run_exp3a.sh](runners/run_exp3a.sh))**:
   - Evaluated token completion budgets of $16,000$, $8,000$, and $4,000$ tokens across 9 open-weights models on $N=200\text{--}203$ Direct Harm cases.
   - *Empirical Diagnostic*: The natural reasoning chains of standard and hybrid compact models conclude well within $1,000\text{--}1,500$ tokens (max lengths: Qwen $\approx 1,121$ tokens; Gemma $\approx 1,293$ tokens; Nano $\approx 1,477$ tokens). Macro-budgets $\ge 4,000$ tokens never truncated reasoning and produced zero statistically significant shift in ASR (variance $\le 1.97\text{ pp}$).
   - *Model Selection*: Only **Nanbeige-4.2-3B** naturally generates extended reasoning traces reaching up to $54,462$ characters ($\sim 13,616$ tokens) with non-zero baseline ASR ($2.47\%$).
2. **Stage 2: Deep Micro-Budget Constraint ([run_exp3a_v3.sh](runners/run_exp3a_v3.sh))**:
   - Evaluated Nanbeige-4.2-3B across fine-grained budgets of $2,000$, $1,000$, $512$, and $256$ tokens on the full $1,054$ cases ($510$ Direct Harm + $544$ Data Stealing) to capture both single-step and two-step exfiltration attacks.

### C. Hardware Infrastructure & Orchestration
* **Hardware Accelerator**: Dedicated NVIDIA RTX A6000 GPU (48GB GDDR6 VRAM with ECC).
* **Serving Backend**: Local `vLLM` instance serving `Nanbeige-4.2-3B` at `http://localhost:8001/v1`.
* **Execution Parameters**:
  - Sampling Temperature: $T = 0.0$ (Strict greedy decoding for determinism).
  - Maximum Completion Tokens: Clamped via the vLLM API parameter `--max_tokens {2000, 1000, 512, 256}`.
  - Concurrency: 16 parallel threads querying the local vLLM instance.
  - Prompt Template: Standard unmodified ReAct prompt ([agent_prompts.py](../00_Baseline_Sweep/prompts/injecagent/agent_prompts.py)), isolating the effect entirely to generation length.

### D. Mathematical Formulations & Tradeoff Metrics

#### 1. Security Gain
$$\text{Security Gain} = \text{ASR}_{\text{baseline}} - \text{ASR}_{\text{budget}}$$

#### 2. Utility Cost (Syntax Collapse Rate)
$$\text{Utility Cost} = \text{Invalid Rate}_{\text{budget}} - \text{Invalid Rate}_{\text{baseline}}$$

#### 3. Matched-Pair Contingency Matrix
Every case $i \in \{1, \dots, 1054\}$ is mapped to a paired transition $(y_{\text{base}}, y_{\text{budget}}) \in \{\text{succ}, \text{fail}\}^2$:
$$\text{Net Flipped Attacks} = N_{(\text{succ} \rightarrow \text{fail})} - N_{(\text{fail} \rightarrow \text{succ})}$$

### E. Methodology-Related Files Directory

| File / Directory | Description & Function |
| :--- | :--- |
| [runners/run_exp3a.sh](runners/run_exp3a.sh) | Stage 1 master orchestration script running 16k/8k/4k token sweeps across all models. |
| [runners/run_exp3a_v2.sh](runners/run_exp3a_v2.sh) | Stage 1 diagnostic logging and validation script. |
| [runners/run_exp3a_v3.sh](runners/run_exp3a_v3.sh) | Stage 2 fine-grained budget constraint runner (2000, 1000, 512, 256 tokens) on Nanbeige. |
| [analyze_exp3a.py](analyze_exp3a.py) | Auditing script parsing raw execution logs, matched pairs, and dose-response curves. |
| [exp3a_summary.json](exp3a_summary.json) | Complete verified aggregate metrics dictionary across all models and budgets. |
| [raw_results/](raw_results/) | Raw execution traces and summary JSONs organized by model and token budget. |

---

## 2. Verified Empirical Results

All metrics below are audited directly from raw task JSON outputs.

### Table 1: Stage 1 Macro-Budget Invariance ($N=200\text{--}203$ cases per budget)

| Model Name | Param Size | Architecture Type | Budget: 16,000 tokens | Budget: 8,000 tokens | Budget: 4,000 tokens | Invalids (16k / 8k / 4k) | Variance Across Budgets |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3.5-4B** | 4B | Gated DeltaNet Hybrid | **9.00%** (18/200) | **9.00%** (18/200) | **9.00%** (18/200) | 4 / 4 / 5 | **0.00 pp** (Completely Invariant) |
| **Qwen3.5-9B** | 9B | Gated DeltaNet Hybrid | **11.82%** (24/203) | **9.85%** (20/203) | **11.33%** (23/203) | 6 / 8 / 7 | **1.97 pp** (Invariant $\pm 1.97\%$) |
| **Gemma-4-E4B-it** | 4B | Sliding Window Hybrid | **1.50%** (3/200) | **1.50%** (3/200) | **1.50%** (3/200) | 0 / 0 / 0 | **0.00 pp** (Completely Invariant) |
| **Gemma-4-12B-it** | 12B | Sliding Window Hybrid | **2.46%** (5/203) | **2.46%** (5/203) | **1.48%** (3/203) | 1 / 1 / 0 | **0.98 pp** (Invariant $\pm 0.98\%$) |
| **Nemotron-3-Nano-4B**| 4B | Mamba2-Transformer Hybrid | **0.00%** (0/200) | **0.00%** (0/200) | **0.00%** (0/200) | 25 / 25 / 24 | **0.00 pp** (Completely Invariant) |
| **Ministral-3-14B** | 14B | Dense Reasoning | **2.96%** (6/203) | **3.94%** (8/203) | **2.96%** (6/203) | 28 / 17 / 21 | **0.98 pp** (Invariant $\pm 0.98\%$) |
| **Ornith-1.5-9B** | 9B | Standard Dense | **10.34%** (21/203) | **9.85%** (20/203) | **10.34%** (21/203) | 5 / 11 / 11 | **0.49 pp** (Invariant $\pm 0.49\%$) |
| **MiniCPM5-2B** | 2B | Standard Dense | **0.00%** (0/203) | **0.00%** (0/203) | **0.00%** (0/203) | 14 / 14 / 10 | **0.00 pp** (Completely Invariant) |
| **Spark-X2.5-4B** | 4B | Standard Dense | **0.00%** (0/203) | **0.00%** (0/203) | **0.00%** (0/203) | 13 / 10 / 13 | **0.00 pp** (Completely Invariant) |
| **Nanbeige-4.2-3B** | 3B | Looped Transformer (22×2) | *Deep micro-budget sweep (Table 2)* | — | — | — | *Evaluated down to 256 tokens* |

*Observation*: Across all 9 evaluated models spanning diverse architectural paradigms (Gated DeltaNet, Sliding Window, Mamba2, Dense Reasoning, and Standard Dense) and scales from 2B to 14B, scaling completion token budgets between 4,000 and 16,000 tokens produces **zero statistically significant behavioral change** (all variations $\le 1.97\text{ pp}$, with 5 models showing strictly $0.00\text{ pp}$ variance). Attack vulnerability is determined by early semantic framing in initial generation rather than late reasoning accumulation.

---

### Table 2: Stage 2 Deep Micro-Budget Dose-Response (Nanbeige-4.2-3B, $N=1,054$)

| Budget | ASR (All Cases) | ASR (Valid Cases) | Invalid Rate | Avg Reasoning Chars | Max Reasoning Chars | DH ASR (%) | DS ASR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **60,000 (Baseline)** | **2.47%** (26) | 3.32% | 25.62% (270) | 10,453 | 54,462 | 0.59% (3) | 4.23% (23) |
| **2,000 Tokens** | **1.23%** (13) | 4.66% | 73.53% (775) | 6,374 | 15,666 | 0.20% (1) | 2.21% (12) |
| **1,000 Tokens** | **0.95%** (10) | 4.35% | 78.18% (824) | 3,453 | 7,781 | 0.20% (1) | 1.65% (9) |
| **512 Tokens** | **0.28%** (3) | 1.80% | 84.16% (887) | 1,848 | 3,303 | 0.00% (0) | 0.55% (3) |
| **256 Tokens** | **0.00%** (0) | **0.00%** | 96.11% (1,013) | 947 | 1,258 | 0.00% (0) | 0.00% (0) |

*Note*: Average and Maximum Reasoning Characters represent the combined length of model deliberation fields (`thinking` + `thinking_step2`) extracted across all trajectories.

---

### Table 3: Matched-Pair Contingency Transitions vs. Baseline ($N=1,054$)

| Comparison | $\text{Succ} \rightarrow \text{Succ}$ | $\text{Succ} \rightarrow \text{Fail}$ (Defended) | $\text{Fail} \rightarrow \text{Succ}$ (Backfire) | $\text{Fail} \rightarrow \text{Fail}$ | Net Flipped Attacks | Exact Binomial $p$-value | McNemar $\chi^2$ ($p$-value) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline vs. 2,000 Tokens** | 13 | **13** | **0** | 1,028 | **+13** | $p = 2.44 \times 10^{-4}$ | $\chi^2 = 11.08$ ($p = 8.74 \times 10^{-4}$) |
| **Baseline vs. 1,000 Tokens** | 9 | **17** | **1** | 1,027 | **+16** | $p = 1.45 \times 10^{-4}$ | $\chi^2 = 12.50$ ($p = 4.07 \times 10^{-4}$) |
| **Baseline vs. 512 Tokens** | 3 | **23** | **0** | 1,028 | **+23** | $p = 2.38 \times 10^{-7}$ | $\chi^2 = 21.04$ ($p = 4.49 \times 10^{-6}$) |
| **Baseline vs. 256 Tokens** | 0 | **26** | **0** | 1,028 | **+26** | $p = 2.98 \times 10^{-8}$ | $\chi^2 = 24.04$ ($p = 9.44 \times 10^{-7}$) |

*Note*: Statistical significance is calculated using two-sided exact binomial tests on discordant pairs ($SF$ vs. $FS$) with $H_0: p = 0.5$, complemented by McNemar's test with Edwards continuity correction ($\text{df} = 1$). All budget reductions exhibit highly significant net attack suppression ($p < 0.0003$).

---

### Table 4: The Severe Utility-Security Tradeoff

| Budget | Valid Outputs | ASR (Valid Only) | Security Gain | Utility Cost | Tradeoff Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 74.38% | 3.32% | 0.00 pp | 0.00 pp | — |
| **2,000 Tokens** | 26.47% | 4.66% | +1.24 pp | +47.91 pp | $1 : 38.6$ |
| **1,000 Tokens** | 21.82% | 4.35% | +1.52 pp | +52.56 pp | $1 : 34.6$ |
| **512 Tokens** | 15.84% | 1.80% | +2.19 pp | +58.54 pp | $1 : 26.7$ |
| **256 Tokens** | 3.89% | 0.00% | +2.47 pp | +70.49 pp | $1 : 28.5$ |

---

## 3. Key Findings & Mechanistic Insights

1. **Reasoning Truncation is Impractical as a Defense**:
   - While clamping `max_tokens` monotonically eliminates ASR ($2.47\% \rightarrow 0.00\%$), it inflicts **catastrophic utility collapse**: at 256 tokens, $96.11\%$ of agent outputs are invalid syntax crashes.
   - For every $1\text{ pp}$ of security gain, the model sacrifices approximately **$27\text{--}39\text{ pp}$ of task utility** (Tradeoff Ratio between $1:26.7$ and $1:38.6$). Reasoning models cannot separate "reasoning about the task" from "reasoning about the injection"—both share the identical computational budget.
2. **Data Stealing Requires Multi-Turn Cognitive Bandwidth**:
   - Data Stealing ASR drops sharply from $4.23\%$ to $0.55\%$ at 512 tokens and $0.00\%$ at 256 tokens. Exfiltrating data requires multi-turn trajectory planning; truncating generation starves the model of the sequence length necessary to format the outbound payload.
3. **Negligible Backfire Rate (Single Isolated Flip at 1,000 Tokens)**:
   - Across all 4,216 budget-constrained trajectories on Nanbeige, exactly **one single case flipped from $\text{Fail} \rightarrow \text{Succ}$** (`data_stealing_case_535.json` at 1,000 tokens; $FS = 1$, $0.09\%$), while **79 attacks were successfully eliminated** ($SF = 79$). This demonstrates a **$79:1$ defense-to-backfire ratio**, confirming that reasoning truncation does not systematically introduce new attack vulnerabilities.
4. **Transient Concentration Effect at 2,000 Tokens**:
   - ASR among *valid* outputs paradoxically increases from $3.32\%$ to $4.66\%$ at 2,000 tokens (and $4.35\%$ at 1,000 tokens). This occurs because the budget limit preferentially invalidates long, complex failed deliberation trajectories, leaving behind simple, hasty trajectories—including direct compliance.

---

## 4. Paper Placement & Contribution (§5.3)
This experiment appears in Section 5.3 (*Test-Time Compute and Reasoning Capacity*). It provides definitive empirical evidence that while reasoning capacity is mechanically necessary for complex multi-step prompt injection, test-time token clamping is a non-viable defensive strategy due to catastrophic utility destruction.
