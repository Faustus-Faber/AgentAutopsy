# 🔓 Administrative Authority Framing & The Untriggered Default

> **Thesis Chapter Reference**: Chapter 5, §5.4.2 ("Administrative Authority Framing and the Untriggered Default") & Chapter 6, §6.5.2 ("Administrative Authority Framing")  
> **Key Results**: Table 6.5 (Administrative authorization framing on InjecAgent, $N=10{,}707$ Authorize, $N=11{,}418$ Interdict)

---

## 1. Experimental Overview

This experiment evaluates the **"Untriggered Default"** hypothesis: compact agents do not possess an intrinsic attack success rate; rather, their baseline resistance is a manifestation of unprompted action inertia. By systematically alternating the system prompt between two administrative policy framings—**Authorize** (framing tool instructions as legitimate secondary work) versus **Interdict** (framing them as security threats)—we test the causal boundaries of agent compliance with model weights strictly fixed.

<div align="center">
  <img src="../assets/images/f2_authority.png" alt="Authority Escalation Dynamics" width="94%" />
  <p><em>Figure: Panel a illustrates how administrative authority framing shifts attack success further than model selection; Panel b shows the corresponding monotonic collapse of safe utility (S0U1).</em></p>
</div>

---

## 2. Key Empirical Findings

1. ⚡ **339× Policy Dynamic Range**: Across the 11-model cohort on InjecAgent, pooled ASR shifted across a **$42.85\,\text{pp}$ gulf**:
   * **Interdict**: Collapsed pooled ASR to **$0.22\%$** ($-7.87\,\text{pp}$ from baseline $8.09\%$).
   * **Authorize**: Surged pooled ASR to **$43.07\%$** ($+34.98\,\text{pp}$ from baseline $8.09\%$).
2. 📈 **Extreme Model Surges**:
   * `Spark-4B`: Surged from $0.95\% \to \mathbf{70.88\%}$ ($+69.93\,\text{pp}$).
   * `Gemma-4B`: Surged from $7.02\% \to \mathbf{72.69\%}$ ($+65.67\,\text{pp}$).
   * `Qwen-9B`: Surged from $25.14\% \to \mathbf{74.57\%}$ ($+49.43\,\text{pp}$).
3. ⚠️ **Execution Validity Caveats**: Low Authorize ASR in `LFM-2.6B` ($0.00\%$) and `Ministral-14B` ($0.11\%$) was driven by massive invalid tool calling ($724 / 1{,}054$ and $217 / 900$) rather than intentional policy refusal.

---

## 3. Directory Layout

* `analyze_authority_framing.py`: Comprehensive paired analysis and statistical tests.
* `authority_framing_summary.json`: Audited cell-level outcomes across all 11 architectures.
* `raw_results/`: Full raw execution trajectories across `injecagent/` and `agentdojo/`.
* `prompts/`: Verbatim policy framing definitions.
* `runner/`: Multi-threaded test runners for InjecAgent and AgentDojo.
* `EXP_Report.md`: Full technical analysis report.

---

## 4. Instant Reproduction Command

```bash
python analyze_authority_framing.py
```
