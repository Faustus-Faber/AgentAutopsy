#!/usr/bin/env python3
"""
Clean Analysis Script: EXP_5A_Reasoning_Ablation
Evaluates the causal role of Chain-of-Thought (CoT) reasoning by comparing
thinkON vs thinkOFF across Control, Boundary (EXP 2A), Authorize (Reframe),
and Interdict (Anomaly) conditions.

Directly parses all raw case JSON files on disk in raw_results/ and contextually
anchors thinkON comparisons with audited upstream benchmarks (00_Baseline_Sweep,
EXP_2A_Boundary_Guardrail, and EXP_2BFIX_Untriggered_Default).
"""

import os
import json
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_results")

# Upstream paths
BASELINE_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "01_dual_benchmark_baselines", "baseline_summary.json"))
if not os.path.exists(BASELINE_FILE):
    BASELINE_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "00_Baseline_Sweep", "baseline_summary.json"))
EXP2A_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "03_instruction_boundary_guardrails", "exp2a_summary.json"))
if not os.path.exists(EXP2A_FILE):
    EXP2A_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "EXP_2A_Boundary_Guardrail", "exp2a_summary.json"))
EXP2B_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "04_administrative_authority_framing", "exp2bfix_summary.json"))
if not os.path.exists(EXP2B_FILE):
    EXP2B_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "EXP_2BFIX_Untriggered_Default", "exp2bfix_summary.json"))

MODELS = [
    ("qwen9b", "Qwen3.5-9B", "9B", "Gated DeltaNet Hybrid"),
    ("gemma", "Gemma-4-E4B-it", "4B", "Sliding Window Hybrid"),
    ("ministral", "Ministral-3-14B-Instruct-2512", "14B", "Dense Reasoning"),
    ("ornith", "Ornith-1.5-9B", "9B", "Standard Dense"),
    ("gemma12b", "Gemma-4-12B-it", "12B", "Sliding Window Hybrid"),
    ("qwen", "Qwen3.5-4B", "4B", "Gated DeltaNet Hybrid"),
    ("nano", "Nemotron-3-Nano-4B", "4B", "Mamba2-Transformer Hybrid"),
    ("nanbeige", "Nanbeige-4.2-3B", "3B", "Looped Transformer (22×2)")
]

CONDITIONS = [
    ("control", "Control (Baseline)"),
    ("exp2a", "Boundary (EXP 2A)"),
    ("reframe", "Authorize (Reframe)"),
    ("anomaly", "Interdict (Anomaly)")
]

def load_upstream():
    base_data = {}
    exp2a_data = {}
    exp2b_data = {}

    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            base_data = json.load(f)
    if os.path.exists(EXP2A_FILE):
        with open(EXP2A_FILE, "r", encoding="utf-8") as f:
            exp2a_data = json.load(f)
    if os.path.exists(EXP2B_FILE):
        with open(EXP2B_FILE, "r", encoding="utf-8") as f:
            exp2b_data = json.load(f)

    return base_data, exp2a_data, exp2b_data

def parse_case_files(c_dir):
    files = [f for f in glob.glob(os.path.join(c_dir, "*.json")) if not any(x in f for x in ['metrics', 'progress', 'summary', 'run'])]
    tot = len(files)
    succ = 0
    inval = 0
    unsucc = 0
    for cfp in files:
        try:
            with open(cfp, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            ev = d.get("eval")
            if ev == "succ":
                succ += 1
            elif ev == "invalid":
                inval += 1
            elif ev == "unsucc":
                unsucc += 1
        except Exception:
            pass

    asr = (succ / tot * 100) if tot else 0.0
    inval_rate = (inval / tot * 100) if tot else 0.0
    return {
        "total": tot,
        "succ": succ,
        "invalid": inval,
        "unsucc": unsucc,
        "asr": round(asr, 2),
        "inval_rate": round(inval_rate, 2)
    }

def analyze_exp5a():
    print("=" * 115)
    print("EXP 5A: REASONING ABLATION (thinkON vs thinkOFF) AUDITED FACTORIAL PIPELINE")
    print("=" * 115)

    base_data, exp2a_data, exp2b_data = load_upstream()

    parsed_raw = {}
    for mid, name, params, arch in MODELS:
        m_dir = os.path.join(RAW_DIR, mid)
        if not os.path.exists(m_dir):
            continue
        parsed_raw[mid] = {}
        for cond_name in sorted(os.listdir(m_dir)):
            c_dir = os.path.join(m_dir, cond_name)
            if not os.path.isdir(c_dir):
                continue
            parsed_raw[mid][cond_name] = parse_case_files(c_dir)

    # 1. Build Table 1: Factorial Sweep
    table1 = []
    print(f"{'Model':<28} | {'Condition':<20} | {'thinkON ASR':>12} | {'thinkOFF ASR':>13} | {'Shift (OFF - ON)':>18}")
    print("-" * 115)

    for mid, name, params, arch in MODELS:
        m_raw = parsed_raw.get(mid, {})
        for c_key, c_label in CONDITIONS:
            off_key = f"{c_key}_thinkOFF"
            if off_key not in m_raw:
                continue

            off_metrics = m_raw[off_key]
            off_asr = off_metrics["asr"]

            # Ground-truth thinkON baseline resolution
            b_key = f"{mid}_fp16"
            if c_key == "control":
                # Check if matched on-disk thinkON exists (Nanbeige)
                if mid == "nanbeige" and "control_thinkON" in m_raw:
                    on_asr = m_raw["control_thinkON"]["asr"]
                    source_note = f"Matched on-disk control_thinkON (N={m_raw['control_thinkON']['total']}, {m_raw['control_thinkON']['succ']} succ; 00_Baseline is 2.47%)"
                else:
                    on_asr = base_data.get("injecagent", {}).get(b_key, {}).get("asr", 0.0)
                    source_note = "00_Baseline_Sweep FP16 unprompted full cohort"
            elif c_key == "exp2a":
                on_asr = exp2a_data.get("per_model_results", {}).get(b_key, {}).get("exp2a_asr", 0.0)
                source_note = "EXP_2A_Boundary_Guardrail FP16"
            elif c_key == "reframe":
                on_asr = exp2b_data.get("injecagent_per_cell", {}).get(f"{mid}_authorize", {}).get("asr", 0.0)
                source_note = "EXP_2BFIX Authorize (Reframe) audited cohort"
            elif c_key == "anomaly":
                on_asr = exp2b_data.get("injecagent_per_cell", {}).get(f"{mid}_interdict", {}).get("asr", 0.0)
                source_note = "EXP_2BFIX Interdict (Anomaly) audited cohort"
            else:
                on_asr = 0.0
                source_note = "Unknown"

            delta = round(off_asr - on_asr, 2)

            t1_entry = {
                "model_id": mid,
                "model_name": name,
                "architecture": arch,
                "condition_key": c_key,
                "condition_name": c_label,
                "n_off": off_metrics["total"],
                "succ_off": off_metrics["succ"],
                "inval_off": off_metrics["invalid"],
                "thinkOFF_asr": off_asr,
                "thinkON_asr": on_asr,
                "delta_off_minus_on": delta,
                "source_note": source_note
            }
            table1.append(t1_entry)
            print(f"{name:<28} | {c_label:<20} | {on_asr:>11.2f}% | {off_asr:>12.2f}% | {delta:>+17.2f}pp")

    # 2. Build Table 2: Syntactic Formatting Degradation
    print("\n" + "=" * 115)
    print("TABLE 2: SYNTACTIC TOOL FORMATTING DEGRADATION UNDER thinkOFF")
    print("=" * 115)
    print(f"{'Model':<28} | {'Control Inval':<16} | {'Boundary Inval':<16} | {'Authorize Inval':<16} | {'Interdict Inval':<16}")
    print("-" * 115)

    table2 = []
    for mid, name, params, arch in MODELS:
        m_raw = parsed_raw.get(mid, {})
        c_inv = m_raw.get("control_thinkOFF", {})
        b_inv = m_raw.get("exp2a_thinkOFF", {})
        a_inv = m_raw.get("reframe_thinkOFF", {})
        i_inv = m_raw.get("anomaly_thinkOFF", {})

        c_str = f"{c_inv.get('invalid',0)}/{c_inv.get('total',0)} ({c_inv.get('inval_rate',0):.1f}%)" if c_inv else "N/A"
        b_str = f"{b_inv.get('invalid',0)}/{b_inv.get('total',0)} ({b_inv.get('inval_rate',0):.1f}%)" if b_inv else "N/A"
        a_str = f"{a_inv.get('invalid',0)}/{a_inv.get('total',0)} ({a_inv.get('inval_rate',0):.1f}%)" if a_inv else "N/A"
        i_str = f"{i_inv.get('invalid',0)}/{i_inv.get('total',0)} ({i_inv.get('inval_rate',0):.1f}%)" if i_inv else "N/A"

        print(f"{name:<28} | {c_str:<16} | {b_str:<16} | {a_str:<16} | {i_str:<16}")

        table2.append({
            "model_id": mid,
            "model_name": name,
            "architecture": arch,
            "control_invalid": c_inv.get("invalid", 0),
            "control_total": c_inv.get("total", 0),
            "control_inval_rate": c_inv.get("inval_rate", 0.0),
            "boundary_invalid": b_inv.get("invalid", 0),
            "boundary_total": b_inv.get("total", 0),
            "boundary_inval_rate": b_inv.get("inval_rate", 0.0),
            "authorize_invalid": a_inv.get("invalid", 0),
            "authorize_total": a_inv.get("total", 0),
            "authorize_inval_rate": a_inv.get("inval_rate", 0.0),
            "interdict_invalid": i_inv.get("invalid", 0),
            "interdict_total": i_inv.get("total", 0),
            "interdict_inval_rate": i_inv.get("inval_rate", 0.0),
        })

    # Nanbeige full cohort matched comparison
    nb_on = parsed_raw["nanbeige"]["control_thinkON"]
    nb_off = parsed_raw["nanbeige"]["control_thinkOFF"]
    surge_mult = round(nb_off["inval_rate"] / nb_on["inval_rate"], 2) if nb_on["inval_rate"] else 0.0
    print(f"\nNanbeige Matched Comparison: ON={nb_on['invalid']}/{nb_on['total']} ({nb_on['inval_rate']}%) vs OFF={nb_off['invalid']}/{nb_off['total']} ({nb_off['inval_rate']}%) -> {surge_mult}x surge")

    # 3. Compute Authority Surge Interaction (Gamma)
    interaction_metrics = {}
    for mid, name, params, arch in MODELS:
        m_raw = parsed_raw.get(mid, {})
        if "reframe_thinkOFF" in m_raw and "control_thinkOFF" in m_raw:
            off_auth = m_raw["reframe_thinkOFF"]["asr"]
            off_ctrl = m_raw["control_thinkOFF"]["asr"]
            off_surge = round(off_auth - off_ctrl, 2)

            # Sourced ON
            b_key = f"{mid}_fp16"
            if mid == "nanbeige" and "control_thinkON" in m_raw:
                on_ctrl = m_raw["control_thinkON"]["asr"]
            else:
                on_ctrl = base_data.get("injecagent", {}).get(b_key, {}).get("asr", 0.0)
            on_auth = exp2b_data.get("injecagent_per_cell", {}).get(f"{mid}_authorize", {}).get("asr", 0.0)
            on_surge = round(on_auth - on_ctrl, 2)
            gamma = round(off_surge - on_surge, 2)

            interaction_metrics[mid] = {
                "off_surge": off_surge,
                "on_surge": on_surge,
                "gamma_interaction": gamma
            }

    summary = {
        "metadata": {
            "suite": "EXP_5A_Reasoning_Ablation",
            "eval_framework": "InjecAgent (N=100-103 cases per cell, N=1043-1054 for Nanbeige)",
            "evaluated_models": 8,
            "non_reasoning_controls": 3,
            "sampling": "Deterministic Greedy (T=0.0)"
        },
        "table1_factorial_sweep": table1,
        "table2_syntactic_degradation": table2,
        "nanbeige_matched_cohort": {
            "thinkON": nb_on,
            "thinkOFF": nb_off,
            "surge_multiplier": surge_mult
        },
        "interaction_metrics": interaction_metrics
    }

    out_file = os.path.join(BASE_DIR, "exp5a_summary.json")
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n[OK] Audited results serialized to {out_file}")

if __name__ == "__main__":
    analyze_exp5a()
