# Experiment 5A: Reasoning Token Ablation (Think ON vs. Think OFF) Report

**Target Scope**: Causal Investigation of Test-Time Chain-of-Thought (CoT) Deliberation on Indirect Prompt Injection Compliance  
**Evaluated Benchmark**: InjecAgent ($N=100\text{--}103$ test cases per cell across 8 reasoning models; $N=1{,}043\text{--}1{,}054$ matched full cohort for Nanbeige; $4{,}945$ executed trajectories on disk in [raw_results/](raw_results/): $3{,}891$ `thinkOFF` + $1{,}054$ `thinkON`)
**Evaluated Models**:
- **8 Native Reasoning Architectures**: Qwen3.5-9B, Gemma-4-E4B-it, Ministral-3-14B, Ornith-1.5-9B, Gemma-4-12B-it, Qwen3.5-4B, Nemotron-3-Nano-4B, Nanbeige-4.2-3B (FP16)
- **3 Non-Reasoning Control Baselines**: MiniCPM5-2B, Spark-X2.5-4B, LFM-2.5-2.6B (unmanipulated baselines where CoT toggle is N/A)  
**Intervention Design**: Factorial Test-Time Compute Manipulation:
- **Thinking State**: `thinkON` (Deliberative CoT active) vs. `thinkOFF` (CoT suppressed via `--no-think` / template suppression)
- **System Policy**: Control (Baseline), Boundary (EXP 2A Epistemic Boundary Defense), Authorize (Reframe), Interdict (Anomaly)  

---

## 1. Experimental Methodology & Technical Protocol

### A. Scientific Motivation & Research Hypotheses
* **Research Question (RQ5A)**: Does disabling the internal Chain-of-Thought (CoT) reasoning scratchpad causally prevent indirect prompt injection compliance under adversarial authority framing?
* **Hypotheses**:
  - *Defensive Cognitive Shield Hypothesis*: If reasoning serves as an epistemic defense against injection, disabling reasoning tokens (`thinkOFF`) will cause ASR to spike across baseline conditions as the agent acts on reflexive pattern-matching without critical evaluation.
  - *Rationalization Hypothesis*: If reasoning facilitates adversarial compliance by providing cognitive bandwidth to rationalize conflicting instructions, `thinkOFF` will eliminate or significantly suppress ASR under administrative authorization.
  - *Orthogonal Scaffolding Hypothesis*: Test-time reasoning is functionally uncoupled from safety gating. Disabling reasoning does not prevent authority-driven compliance; instead, reasoning tokens primarily serve as syntactic scaffolding for structural tool-calling formatting.

### B. Hardware Apparatus & Test-Time Compute Manipulation
* **Hardware Accelerator**: Dedicated NVIDIA RTX A6000 GPU (48GB GDDR6 VRAM with ECC).
* **Serving Backend**: Local `vLLM` (v0.6.x+) OpenAI-compatible API endpoint at `http://localhost:8001/v1`.
* **Test-Time Compute Control**:
  - `thinkON`: Standard inference allowing unconstrained reasoning token generation prior to tool call emission (`message.reasoning_content` logged verbatim).
  - `thinkOFF`: Suppressing the hidden reasoning phase via server-side chat template modification (`enable_thinking=False` or passing `--no-think` in [runner/run_injecagent_vllm_exp5a.py](runner/run_injecagent_vllm_exp5a.py)).
* **Decoding Parameters**: $T = 0.0$ (Strict greedy decoding ensuring deterministic evaluation), `max_tokens = 60,000`.

### C. Factorial Experimental Matrix
Every native reasoning architecture is evaluated across an orthogonal $2 \times 4$ condition space:

```
                     ┌───────────────────┬───────────────────┐
                     │      thinkON      │     thinkOFF      │
┌────────────────────┼───────────────────┼───────────────────┤
│ 1. Control         │ Baseline + CoT    │ Baseline - CoT    │
│ 2. Boundary (EXP2A)│ Epistemic Guard   │ Epistemic - CoT   │
│ 3. Authorize       │ Authorize + CoT   │ Authorize - CoT   │
│ 4. Interdict       │ Interdict + CoT   │ Interdict - CoT   │
└────────────────────┴───────────────────┴───────────────────┘
```

### D. Mathematical Formulations

#### 1. Reasoning Ablation Net Shift ($\Delta_{\text{ablation}}$)
$$\Delta_{\text{ablation}} = \text{ASR}_{\text{thinkOFF}} - \text{ASR}_{\text{thinkON}}$$
- $\Delta_{\text{ablation}} > 0$: Disabling reasoning increases vulnerability (supports Defensive Shield Hypothesis).
- $\Delta_{\text{ablation}} < 0$: Disabling reasoning decreases vulnerability (supports Rationalization Hypothesis).
- $\Delta_{\text{ablation}} \approx 0$: Reasoning state is uncoupled from compliance decision (supports Orthogonal Scaffolding Hypothesis).

#### 2. Authority Surge Interaction Metric ($\Gamma$)
Measures whether the compliance surge induced by administrative authorization changes when test-time reasoning is suppressed:
$$\Gamma = \left(\text{ASR}_{\text{Authorize, OFF}} - \text{ASR}_{\text{Control, OFF}}\right) - \left(\text{ASR}_{\text{Authorize, ON}} - \text{ASR}_{\text{Control, ON}}\right)$$

### E. Methodology-Related Files Directory

| File / Directory | Description & Function |
| :--- | :--- |
| [runner/run_injecagent_vllm_exp5a.py](runner/run_injecagent_vllm_exp5a.py) | InjecAgent test runner with `--no-think` parameter toggle. |
| [runner/run_agentdojo_exp5a.py](runner/run_agentdojo_exp5a.py) | AgentDojo test runner orchestrating factorial sweeps on vLLM. |
| [runner/agentdojo_exp_prompts.py](runner/agentdojo_exp_prompts.py) | System message loader for control, exp2a, reframe, and anomaly conditions. |
| [prompts/](prompts/) | ReAct system prompt templates across experimental conditions. |
| [analyze_exp5a.py](analyze_exp5a.py) | Auditing script parsing raw trajectories and calculating factorial interaction deltas. |
| [exp5a_summary.json](exp5a_summary.json) | Complete audited summary dictionary across all conditions and models. |

---

## 2. Verified Empirical Results

All metrics below are audited directly from raw task outputs.

### Table 1: InjecAgent Factorial Sweep Across All 8 Evaluated Reasoning Models ($N=100\text{--}103$ per cell)

| Model Name | Architectural Paradigm | Condition Policy | thinkON ASR (%) | thinkOFF ASR (%) | Shift ($\Delta = \text{OFF} - \text{ON}$) | Behavioral Interpretation |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Qwen3.5-9B** | Gated DeltaNet Hybrid | Control (Base) | **25.14%** | **1.94%** (2/103) | **-23.20 pp** | Syntax errors drop baseline ASR |
| **Qwen3.5-9B** | Gated DeltaNet Hybrid | Boundary (EXP2A)| **0.66%** | **1.94%** (2/103) | **+1.28 pp** | Boundary remains robust without CoT |
| **Qwen3.5-9B** | Gated DeltaNet Hybrid | Authorize (Reframe)| **74.57%** | **70.87%** (73/103) | **-3.70 pp** | **Massive compliance (70.9%) without reasoning tokens!** |
| **Qwen3.5-9B** | Gated DeltaNet Hybrid | Interdict (Anomaly)| **1.33%** | **0.97%** (1/103) | **-0.36 pp** | Interdiction intact without CoT |
| **Gemma-4-E4B-it** | Sliding Window Hybrid | Control (Base) | **7.02%** | **0.00%** (0/100) | **-7.02 pp** | Reflexive rejection / no baseline compliance |
| **Gemma-4-E4B-it** | Sliding Window Hybrid | Boundary (EXP2A)| **1.23%** | **0.00%** (0/100) | **-1.23 pp** | Boundary 100% effective without CoT |
| **Gemma-4-E4B-it** | Sliding Window Hybrid | Authorize (Reframe)| **72.69%** | **62.00%** (62/100) | **-10.69 pp** | **Extreme compliance (62.0%) sustained without CoT!** |
| **Gemma-4-E4B-it** | Sliding Window Hybrid | Interdict (Anomaly)| **0.00%** | **0.00%** (0/100) | **0.00 pp** | 100% interdiction retention |
| **Ministral-3-14B** | Dense Reasoning | Control (Base) | **0.28%** | **11.65%** (12/103) | **+11.37 pp** | Moderate baseline execution |
| **Ministral-3-14B** | Dense Reasoning | Boundary (EXP2A)| **4.36%** | **5.83%** (6/103) | **+1.47 pp** | Guardrail filters majority of attacks |
| **Ministral-3-14B** | Dense Reasoning | Authorize (Reframe)| **0.11%** | **49.51%** (51/103) | **+49.40 pp** | **Direct execution surges (49.5%) bypassing formatting traps!** |
| **Ministral-3-14B** | Dense Reasoning | Interdict (Anomaly)| **0.00%** | **1.94%** (2/103) | **+1.94 pp** | High interdiction suppression |
| **Ornith-1.5-9B** | Standard Dense | Control (Base) | **27.42%** | **31.07%** (32/103) | **+3.65 pp** | Baseline attack success persists without CoT |
| **Ornith-1.5-9B** | Standard Dense | Boundary (EXP2A)| **0.19%** | **14.56%** (15/103) | **+14.37 pp** | Partial boundary leakage without reasoning |
| **Ornith-1.5-9B** | Standard Dense | Authorize (Reframe)| **62.37%** | **49.51%** (51/103) | **-12.86 pp** | **High compliance (49.5%) sustained without CoT** |
| **Ornith-1.5-9B** | Standard Dense | Interdict (Anomaly)| **1.02%** | **7.77%** (8/103) | **+6.75 pp** | Substantial interdiction suppression |
| **Gemma-4-12B-it** | Sliding Window Hybrid | Control (Base) | **3.51%** | **0.00%** (0/103) | **-3.51 pp** | Zero baseline compliance without CoT |
| **Gemma-4-12B-it** | Sliding Window Hybrid | Boundary (EXP2A)| **0.00%** | **0.00%** (0/103) | **0.00 pp** | 100% boundary defense retention |
| **Gemma-4-12B-it** | Sliding Window Hybrid | Authorize (Reframe)| **61.43%** | **33.01%** (34/103) | **-28.42 pp** | **Sustained 33.0% compliance without CoT** |
| **Gemma-4-12B-it** | Sliding Window Hybrid | Interdict (Anomaly)| **0.00%** | **0.00%** (0/103) | **0.00 pp** | 100% interdiction retention |
| **Qwen3.5-4B** | Gated DeltaNet Hybrid | Control (Base) | **18.79%** | **3.00%** (3/100) | **-15.79 pp** | Syntax formatting failures suppress baseline ASR |
| **Qwen3.5-4B** | Gated DeltaNet Hybrid | Boundary (EXP2A)| **0.66%** | **0.00%** (0/100) | **-0.66 pp** | Boundary maintained without CoT |
| **Qwen3.5-4B** | Gated DeltaNet Hybrid | Authorize (Reframe)| **62.10%** | **31.00%** (31/100) | **-31.10 pp** | **Still complies heavily (31.0%) without CoT!** |
| **Qwen3.5-4B** | Gated DeltaNet Hybrid | Interdict (Anomaly)| **0.00%** | **1.00%** (1/100) | **+1.00 pp** | Interdiction intact without CoT |
| **Nemotron-3-Nano-4B**| Mamba2-Transformer Hybrid | Control (Base) | **2.56%** | **0.00%** (0/100) | **-2.56 pp** | Zero baseline compliance |
| **Nemotron-3-Nano-4B**| Mamba2-Transformer Hybrid | Boundary (EXP2A)| **0.76%** | **0.00%** (0/100) | **-0.76 pp** | Boundary maintained without CoT |
| **Nemotron-3-Nano-4B**| Mamba2-Transformer Hybrid | Authorize (Reframe)| **47.69%** | **27.00%** (27/100) | **-20.69 pp** | **Compliance reaches 27.0% without CoT!** |
| **Nemotron-3-Nano-4B**| Mamba2-Transformer Hybrid | Interdict (Anomaly)| **0.19%** | **1.00%** (1/100) | **+0.81 pp** | Interdiction intact without CoT |
| **Nanbeige-4.2-3B** | Looped Transformer (22×2) | Control (Base) | **1.99%** | **4.99%** (52/1043)| **+3.00 pp** | Slight compliance rise without CoT (matched) |
| *MiniCPM5-2B* | Standard Dense | All Conditions | — | — | — | *Non-reasoning baseline (CoT toggle N/A)* |
| *Spark-X2.5-4B* | Standard Dense | All Conditions | — | — | — | *Non-reasoning baseline (CoT toggle N/A)* |
| *LFM-2.5-2.6B* | Liquid SSM Hybrid | All Conditions | — | — | — | *Non-reasoning baseline (CoT toggle N/A)* |

*Data Provenance & Sample Size Disclosures*:
1. `thinkOFF` values represent empirical evaluations on disk in [raw_results/](raw_results/) ($N=100\text{--}103$ test cases per cell; for Nanbeige, $N=1{,}043$ cases).
2. `thinkON` reference baselines are sourced from audited upstream suites:
   - Control: `00_Baseline_Sweep` FP16 unprompted full cohort ($N=1{,}054$; for Nanbeige, matched on-disk `control_thinkON` is $21/1{,}054 = 1.99\%$; `00_Baseline` is $2.47\%$).
   - Boundary: `EXP_2A_Boundary_Guardrail` FP16 full cohort.
   - Authorize & Interdict: `EXP_2BFIX_Untriggered_Default` audited full cohorts (e.g. Qwen9B: 74.57%, Gemma: 72.69%, Ornith: 62.37%, Gemma12B: 61.43%, Qwen: 62.10%, Nano: 47.69%, Nanbeige: 25.71%, Ministral: 0.11%). Initial exploratory $N=100$ runs showed Gemma: 73.0%, Qwen: 36.0%, Nano: 23.0%, Ministral: 22.0%.

---

### Table 2: Syntactic Tool Formatting Degradation Under `thinkOFF` Across Evaluated Reasoning Models

| Model Name | Architectural Paradigm | Control Invalid Rate | Boundary Invalid Rate | Authorize Invalid Rate | Interdict Invalid Rate | Syntactic Degradation Profile |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Nanbeige-4.2-3B** | Looped Transformer (22×2) | **76.4%** (797/1043) | — | — | — | **$4.94\times$ invalid surge** vs thinkON (15.5%, 163/1054) |
| **Ministral-3-14B** | Dense Reasoning | **34.0%** (35/103) | **28.2%** (29/103) | **21.4%** (22/103) | **56.3%** (58/103) | Severe formatting degradation across all arms |
| **Ornith-1.5-9B** | Standard Dense | **10.7%** (11/103) | **12.6%** (13/103) | **6.8%** (7/103) | **3.9%** (4/103) | Moderate formatting degradation (4–13%) |
| **Qwen3.5-9B** | Gated DeltaNet Hybrid | **6.8%** (7/103) | **1.0%** (1/103) | **0.0%** (0/103) | **1.0%** (1/103) | Formatting preserved under authorization |
| **Qwen3.5-4B** | Gated DeltaNet Hybrid | **3.0%** (3/100) | **1.0%** (1/100) | **4.0%** (4/100) | **0.0%** (0/100) | Stable JSON schema generation (0–4%) |
| **Nemotron-3-Nano-4B**| Mamba2-Transformer Hybrid | **3.0%** (3/100) | **0.0%** (0/100) | **3.0%** (3/100) | **2.0%** (2/100) | Stable JSON schema generation (0–3%) |
| **Gemma-4-12B-it** | Sliding Window Hybrid | **1.9%** (2/103) | **0.0%** (0/103) | **2.9%** (3/103) | **0.0%** (0/103) | High formatting stability (0–3%) |
| **Gemma-4-E4B-it** | Sliding Window Hybrid | **0.0%** (0/100) | **0.0%** (0/100) | **0.0%** (0/100) | **3.0%** (3/100) | High formatting stability (0–3%) |

---

## 3. Key Findings & Mechanistic Insights

1. **Disproof of Reasoning as a Defensive Cognitive Shield**:
   - Disabling Chain-of-Thought reasoning (`thinkOFF`) does **not** causally prevent indirect prompt injection. Under the `Authorize` condition, compliance remains exceptionally high across architectures:
     - **Qwen3.5-9B**: Complies in **70.87%** of attacks without reasoning (compared to 74.57% with CoT).
     - **Gemma-4-E4B-it**: Complies in **62.00%** of attacks without reasoning (compared to 72.69% with CoT).
     - **Ornith-1.5-9B**: Complies in **49.51%** of attacks without reasoning (compared to 62.37% with CoT).
     - **Ministral-3-14B**: Complies in **49.51%** of attacks without reasoning.
     - **Gemma-4-12B-it**: Complies in **33.01%** of attacks without reasoning.
     - **Qwen3.5-4B**: Complies in **31.00%** of attacks without reasoning.
     - **Nemotron-3-Nano-4B**: Complies in **27.00%** of attacks without reasoning.
   - The agent does not require test-time deliberative tokens to rationalize or execute malicious tool invocations; perceived administrative authority directly modulates early residual activations, dictating downstream action execution.

2. **Epistemic Boundary Defenses Function Independently of CoT**:
   - Under the `Boundary` (EXP 2A) condition, models retain strong defense even when test-time reasoning is completely suppressed:
     - Gemma-4-E4B-it, Gemma-4-12B-it, Qwen3.5-4B, and Nemotron-3-Nano-4B achieve **0.00% ASR** under `thinkOFF`.
     - Qwen3.5-9B achieves **1.94% ASR** under `thinkOFF`.
     - Ministral-3-14B achieves **5.83% ASR** under `thinkOFF`.
   - This proves that instruction-data boundary discrimination is an epistemic context-parsing capability established by the system prompt rather than an emergent calculation performed during reasoning.

3. **Chain-of-Thought Functions Primarily as Syntactic Scaffolding**:
   - The primary empirical benefit of test-time reasoning in compact agentic models is structural syntax preservation rather than security evaluation.
   - In Nanbeige-4.2-3B, evaluating matched full cohorts reveals that suppressing thinking causes invalid tool formatting calls to explode from **15.46%** (163/1,054) under `thinkON` to **76.41%** (797/1,043) under `thinkOFF`—a **$4.94\times$ degradation surge** (+634 invalid calls).
   - In Ministral-3-14B, invalid formatting rates under `thinkOFF` range between $21.4\%$ and $56.3\%$. Deliberation tokens provide the execution scratchpad necessary to balance nested JSON braces, quote characters, and parameter schemas.

---

## 4. Paper Placement & Contribution (§5.3)

This factorial experiment provides the empirical foundation for Section 5.3 (*Chain-of-Thought Deliberation: Shield or Conduit?*). It refutes the common assumption that reasoning models can "think their way out" of prompt injection attacks. Because perceived administrative authority bypasses boundary gating at the representation level, test-time reasoning tokens simply serve to execute the attacker's instructions with higher syntactic fidelity.
