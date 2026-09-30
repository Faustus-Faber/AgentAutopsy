# 🧠 Test-Time Compute Manipulation: Reasoning-Token Ablation

> **Thesis Chapter Reference**: Chapter 5, §5.4.4 ("Test-Time Compute Manipulation (2x4 Factorial Design)") & Chapter 6, §6.5.5 ("Reasoning-Token Ablation")  
> **Key Results**: Table 6.8 (Reasoning-token ablation outcomes)

---

## 1. Experimental Overview

To investigate whether test-time Chain-of-Thought (CoT) reasoning acts as a **Defensive Cognitive Shield** (preventing injection), a **Rationalization Engine** (facilitating compliance), or **Orthogonal Scaffolding** (tool format scaffolding), we implement a $2 \times 4$ factorial design:
* **Thinking State**: `thinkON` (deliberative CoT active) vs. `thinkOFF` (CoT suppressed via chat template modification).
* **System Policy**: Control (Baseline), Boundary Guardrail, Authorize, Interdict.

The evaluation covers 28 model-policy cells (7 reasoning models $\times$ 4 policies; $100\text{--}103$ matched Direct Harm cases per cell) plus a dedicated 1,043-case control comparison on `Nanbeige-3B`.

---

## 2. Key Empirical Findings

1. ❌ **Disabling Reasoning Never Made Agents Safer**: Across all 28 cells, `thinkOFF` never produced a statistically significant reduction in attack success rate.
2. 🚨 **Reasoning Suppression Heightened Vulnerability**: In 8 of the 28 cells, attack success **increased significantly** when reasoning was disabled:
   * `Ministral-14B`: Attack success rose from $0/103 \to 51/103$ ($49.5\%$, paired $p < 0.001$).
   * `Ornith-9B`: Baseline success rose from $4.9\% \to 31.1\%$ ($p < 10^{-5}$).
   * `Qwen-9B`: Under Authorize, compliance surged from $43.7\% \to 70.9\%$ ($p < 10^{-4}$).
   * `Nanbeige-3B`: Control successes rose from $18 \to 52$ ($p < 10^{-5}$).
3. 🎯 **Scientific Implication**:
   * Refutes the **Rationalization Hypothesis** (removing reasoning never stopped compliance).
   * Supports the **Defensive Hypothesis** in models where reasoning suppression unlocked reflex compliance (`Ornith-9B`, `Qwen-9B`, `Gemma-4B`).
   * Supports the **Orthogonality Hypothesis** in cells with negligible shifts, showing reasoning serves primarily as syntactic scaffolding.

---

## 3. Directory Layout

* `analyze_reasoning_ablation.py`: Evaluates $\Delta\text{ASR} = \text{ASR}(\text{thinkOFF}) - \text{ASR}(\text{thinkON})$ via paired McNemar tests.
* `reasoning_ablation_summary.json`: Audited cell-level outcomes.
* `raw_results/`: Full raw execution trajectories across 8 reasoning models under thinkON and thinkOFF regimes ($4{,}945$ executed trajectories).
* `prompts/`: System prompts for the 4 policy conditions.
* `runner/`: Execution runner controlling test-time compute toggles.
* `EXP_Report.md`: Full empirical report.

---

## 4. Instant Reproduction Command

```bash
python analyze_reasoning_ablation.py
```
