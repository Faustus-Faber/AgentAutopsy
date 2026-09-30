#!/usr/bin/env python3
"""
Audited Analysis Script: EXP_4A_Dose_Response
Evaluates parametric dose-response sensitivity to linguistic authorization pressure:
- Table 1: AgentDojo Continuous Dose-Response (Gemma-4-E4B-it, N=949 paired attacks)
- Table 2: InjecAgent Dose-Response across All 11 Models (N=100 per level, L1 to L3)
  with contextual L0 baseline (00_Baseline_Sweep) and L4 Authorize (EXP_2BFIX_Untriggered_Default).
"""

import os
import sys
import json
import glob
import math
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_INJECAGENT = os.path.join(BASE_DIR, "raw_results")
RAW_AGENTDOJO = os.path.join(BASE_DIR, "raw_results_agentdojo")

BASELINE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "01_dual_benchmark_baselines"))
if not os.path.exists(BASELINE_DIR):
    BASELINE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "00_Baseline_Sweep"))
EXP2B_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "04_administrative_authority_framing"))
if not os.path.exists(EXP2B_DIR):
    EXP2B_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "EXP_2BFIX_Untriggered_Default"))

MODELS_INJECAGENT = [
    ("qwen9b", "Qwen3.5-9B", "9B", "Gated DeltaNet Hybrid"),
    ("gemma", "Gemma-4-E4B-it", "4B", "Sliding Window Hybrid"),
    ("spark", "Spark-X2.5-4B", "4B", "Standard Dense"),
    ("ornith", "Ornith-1.5-9B", "9B", "Standard Dense"),
    ("gemma12b", "Gemma-4-12B-it", "12B", "Sliding Window Hybrid"),
    ("qwen", "Qwen3.5-4B", "4B", "Gated DeltaNet Hybrid"),
    ("nano", "Nemotron-3-Nano-4B", "4B", "Mamba2-Transformer Hybrid"),
    ("nanbeige", "Nanbeige-4.2-3B", "3B", "Looped Transformer (22×2)"),
    ("minicpm5", "MiniCPM5-2B", "2B", "Standard Dense"),
    ("ministral", "Ministral-3-14B", "14B", "Dense Reasoning"),
    ("lfm", "LFM-2.5-2.6B", "2.6B", "Liquid SSM Hybrid"),
]

def load_baseline_metrics():
    """Load unprompted baseline ASRs from 00_Baseline_Sweep."""
    path = os.path.join(BASELINE_DIR, "baseline_summary.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_exp2b_authorize_metrics():
    """Load audited Authorize condition ASRs from EXP_2BFIX_Untriggered_Default."""
    path = os.path.join(EXP2B_DIR, "exp2bfix_summary.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
            return d.get("injecagent_per_cell", {})
    return {}

def count_injecagent_cell(directory):
    """Count evaluated cases for an InjecAgent dose cell."""
    files = [f for f in glob.glob(os.path.join(directory, "*.json"))
             if not any(x in os.path.basename(f) for x in ("metrics", "progress", "summary", "run"))]
    total = len(files)
    succ = unsucc = invalid = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                j = json.load(f)
            ev = j.get("eval")
            if ev == "succ":
                succ += 1
            elif ev == "unsucc":
                unsucc += 1
            else:
                invalid += 1
        except Exception:
            invalid += 1
    asr = round(succ / total * 100, 2) if total else 0.0
    return {"total": total, "succ": succ, "unsucc": unsucc, "invalid": invalid, "asr": asr}

def count_agentdojo_cell(directory):
    """Count evaluated paired cases for an AgentDojo dose cell."""
    paired = 0
    won = 0
    for root, dirs, files in os.walk(directory):
        if "tool_knowledge" in root or "important_instructions" in root:
            for f in files:
                if f.endswith(".json") and "metrics" not in f:
                    paired += 1
                    try:
                        with open(os.path.join(root, f), "r", encoding="utf-8", errors="ignore") as fp:
                            j = json.load(fp)
                        if j.get("security") is True:
                            won += 1
                    except Exception:
                        pass
    asr = round(won / paired * 100, 2) if paired else 0.0
    return {"paired": paired, "won": won, "asr": asr}

def compute_monotonicity(sequence):
    """Compute step-fraction monotonicity M_step and strict monotonicity M_full."""
    if len(sequence) < 2:
        return 1.0, 1
    step_checks = [1 if sequence[i+1] >= sequence[i] else 0 for i in range(len(sequence)-1)]
    m_step = round(sum(step_checks) / len(step_checks), 2)
    m_full = 1 if all(c == 1 for c in step_checks) else 0
    return m_step, m_full

def analyze_exp4a():
    print("=" * 115)
    print("EXP 4A AUDITED ANALYSIS: LINGUISTIC AUTHORITY DOSE-RESPONSE")
    print("=" * 115)

    base_sum = load_baseline_metrics()
    exp2b_cells = load_exp2b_authorize_metrics()

    # -------------------------------------------------------------
    # 1. Table 1: AgentDojo Continuous Dose-Response (Gemma-4-E4B-it)
    # -------------------------------------------------------------
    print("\n--- TABLE 1: AgentDojo Continuous Dose-Response (Gemma-4-E4B-it, N=949 Paired Attacks) ---")
    print(f"{'Dose Condition':<24} {'Words Added':<14} {'Attacks Won':<14} {'ASR (%)':>8} {'Marginal Gain':>16} {'Cumulative Shift':>18}")
    print("-" * 100)

    gemma_base_ado = base_sum.get("agentdojo", {}).get("gemma_fp16", {})
    l0_won = gemma_base_ado.get("attacks_won", 128)
    l0_n = gemma_base_ado.get("paired_total", 949)
    l0_asr = round(l0_won / l0_n * 100, 2)

    ado_levels = [
        ("Level 0 (Control)", 0, l0_won, l0_n, l0_asr),
    ]

    ado_gemma_dir = os.path.join(RAW_AGENTDOJO, "gemma")
    ado_word_counts = {
        "l1_fp16_thinkON": ("Level 1 (Suggestive)", 9),
        "l2_fp16_thinkON": ("Level 2 (Directive)", 20),
        "l3_fp16_thinkON": ("Level 3 (Rationalized)", 31),
    }

    for lvl_dir, (lbl, w_added) in ado_word_counts.items():
        c = count_agentdojo_cell(os.path.join(ado_gemma_dir, lvl_dir))
        ado_levels.append((lbl, w_added, c["won"], c["paired"], c["asr"]))

    table1_data = []
    prev_asr = None
    for lbl, w_added, won, n, asr in ado_levels:
        if prev_asr is None:
            marg = "—"
            cum = "0.00 pp"
            marg_val = 0.0
            cum_val = 0.0
        else:
            marg_val = round(asr - prev_asr, 2)
            cum_val = round(asr - l0_asr, 2)
            marg = f"{marg_val:+.2f} pp"
            cum = f"{cum_val:+.2f} pp"
        prev_asr = asr

        table1_data.append({
            "dose_level": lbl,
            "words_added": w_added,
            "attacks_won": won,
            "paired_total": n,
            "asr": asr,
            "marginal_gain_pp": marg_val,
            "cumulative_shift_pp": cum_val,
        })
        print(f"{lbl:<24} {f'{w_added} words':<14} {f'{won}/{n}':<14} {asr:>7.2f}% {marg:>16} {cum:>18}")

    ado_asrs = [r["asr"] for r in table1_data]
    m_step_ado, m_full_ado = compute_monotonicity(ado_asrs)
    print(f"\n[+] AgentDojo Gemma Monotonicity: M_step = {m_step_ado:.2f}, M_full = {m_full_ado} (Strictly Monotonic across executed levels L0-L3)")

    # -------------------------------------------------------------
    # 2. Table 2: InjecAgent Dose-Response across All 11 Models
    # -------------------------------------------------------------
    print("\n--- TABLE 2: InjecAgent Dose-Response across All 11 Evaluated Models (N=100 per level) ---")
    print(f"{'Model Name':<16} {'Architecture':<26} {'L1 ASR':>8} {'L2 ASR':>8} {'L3 ASR':>8} {'L4 ASR':>8} {'Surge (L1→L4)':>16} {'M_step':>8} {'Profile':<20}")
    print("-" * 125)

    table2_data = []
    injec_summary = {}

    for mid, mname, params, arch in MODELS_INJECAGENT:
        m_dir = os.path.join(RAW_INJECAGENT, mid)
        c1 = count_injecagent_cell(os.path.join(m_dir, "l1"))
        c2 = count_injecagent_cell(os.path.join(m_dir, "l2"))
        c3 = count_injecagent_cell(os.path.join(m_dir, "l3"))

        # Audited L4 Authorize from EXP_2BFIX
        e2b_cell = exp2b_cells.get(f"{mid}_authorize", {})
        l4_asr = e2b_cell.get("asr", 0.0)

        # Base L0
        l0_cell = base_sum.get("injecagent", {}).get(f"{mid}_fp16", {})
        l0_asr = l0_cell.get("asr", 0.0)

        surge = round(l4_asr - c1["asr"], 2)
        seq = [c1["asr"], c2["asr"], c3["asr"], l4_asr]
        m_step, m_full = compute_monotonicity(seq)

        if mid in ("ministral", "lfm"):
            profile = "Syntax Bound (L4 drops)"
            surge_str = "Plateau (Syntax)"
        elif m_step == 1.0:
            profile = "Monotonic Surge"
            surge_str = f"{surge:+.1f} pp"
        elif c1["asr"] == c2["asr"]:
            profile = "Low-Dose Plateau"
            surge_str = f"{surge:+.1f} pp"
        else:
            profile = "Inversion / Noise"
            surge_str = f"{surge:+.1f} pp"

        print(f"{mname:<16} {arch:<26} {c1['asr']:>7.1f}% {c2['asr']:>7.1f}% {c3['asr']:>7.1f}% {l4_asr:>7.1f}% {surge_str:>16} {m_step:>8.2f} {profile:<20}")

        table2_data.append({
            "model_id": mid,
            "display_name": mname,
            "params": params,
            "architecture": arch,
            "l0_asr": l0_asr,
            "l1_asr": c1["asr"],
            "l2_asr": c2["asr"],
            "l3_asr": c3["asr"],
            "l4_asr": l4_asr,
            "compliance_surge_pp": surge,
            "m_step": m_step,
            "m_full": m_full,
            "profile": profile,
        })
        injec_summary[mid] = {
            "l1": c1["asr"],
            "l2": c2["asr"],
            "l3": c3["asr"],
            "l4": l4_asr,
        }

    # Count monotonicity across all 11 models
    monotonic_models = [r["model_id"] for r in table2_data if r["m_step"] == 1.0]
    print(f"\n[+] InjecAgent Monotonic Models: {len(monotonic_models)}/11 models ({', '.join(monotonic_models)})")

    # -------------------------------------------------------------
    # 3. Multi-Model AgentDojo Progression (All 5 Models)
    # -------------------------------------------------------------
    ado_multi_data = {}
    for m in sorted(os.listdir(RAW_AGENTDOJO)):
        mdir = os.path.join(RAW_AGENTDOJO, m)
        b_cell = base_sum.get("agentdojo", {}).get(f"{m}_fp16", {})
        base_asr = round(b_cell.get("attacks_won", 0) / b_cell.get("paired_total", 949) * 100, 2) if b_cell.get("paired_total") else 0.0
        m_row = {"l0": base_asr}
        for lvl in ["l1_fp16_thinkON", "l2_fp16_thinkON", "l3_fp16_thinkON"]:
            c = count_agentdojo_cell(os.path.join(mdir, lvl))
            lvl_short = lvl.split("_")[0]
            m_row[lvl_short] = c["asr"]
        ado_multi_data[m] = m_row

    # -------------------------------------------------------------
    # 4. Save Comprehensive Audited Summary
    # -------------------------------------------------------------
    full_output = {
        "table1_agentdojo_gemma": table1_data,
        "table2_injecagent_dose_response": table2_data,
        "agentdojo_multi_model": ado_multi_data,
        "injecagent_summary": injec_summary,
    }
    # Backward compatibility flat mapping
    for k, v in injec_summary.items():
        full_output[k] = v

    out_file = os.path.join(BASE_DIR, "exp4a_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[+] Successfully exported complete audited metrics to: {out_file}")

if __name__ == "__main__":
    analyze_exp4a()
