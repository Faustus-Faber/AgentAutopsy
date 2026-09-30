# ⏱️ Reasoning Budget Constraints & Micro-Clamping

> **Thesis Chapter Reference**: Chapter 5, §5.4.5 ("Context Budgeting and Cross-Precision Serving Dynamics") & Chapter 6, §6.5.4 ("Reasoning Budget and Context Limits")  
> **Key Results**: Table 6.7 (Token budget scaling and micro-clamping outcomes)

---

## 1. Experimental Overview

This experiment investigates whether Chain-of-Thought (CoT) reasoning engagement is a causal driver of prompt-injection compliance. We evaluate two tiers of budget constraints on InjecAgent:
* **Macro-Budget Sweeps**: Generation budgets clamped to $16{,}000$, $8{,}000$, and $4{,}000$ tokens across 9 models.
* **Micro-Budget Clamping**: Tight limits of $2{,}000$, $1{,}000$, $512$, and $256$ tokens on `Nanbeige-3B` ($N=1{,}054$ cases).

---

## 2. Key Empirical Findings

1. 📏 **Macro-Budget Invariance**: Standard compact models naturally complete reasoning within $1{,}000\text{--}1{,}500$ tokens. Truncation budgets $\ge 4{,}000$ tokens changed ASR by at most $0.49\,\text{pp}$, confirming that macro context ceilings do not alter attack dynamics.
2. 💥 **Micro-Clamping Induces Syntax Collapse**:
   * On `Nanbeige-3B`, clamping max tokens to 256 dropped ASR from $2.47\%$ to **$0.00\%$** (exact $p = 2.98 \times 10^{-8}$).
   * However, this was **not** a defense: **$96.11\%$ of executions crashed with syntax errors ($1{,}013 / 1{,}054$ invalid tool calls)** due to token truncation mid-JSON.
   * Clamping enforces apparent safety solely by crippling agent functionality.

---

## 3. Directory Layout

* `analyze_reasoning_budget.py`: Calculates ASR shifts and syntax collapse rates.
* `reasoning_budget_summary.json`: Multi-budget clamping outcomes.
* `raw_results/`: Full raw execution trajectories across 10 evaluated models for macro and micro budget regimes.
* `runners/`: vLLM shell runners for macro and micro budget sweeps.
* `EXP_Report.md`: Full empirical report.

---

## 4. Instant Reproduction Command

```bash
python analyze_reasoning_budget.py
```
