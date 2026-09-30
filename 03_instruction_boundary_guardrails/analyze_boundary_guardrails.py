#!/usr/bin/env python3
"""
Clean Analysis Script: EXP_2A_Boundary_Guardrail
Evaluates the effect of Instruction Source Boundary Guardrails on InjecAgent (N=1,054)
comparing baseline ASR (from 00_Baseline_Sweep) vs EXP 2A guardrail ASR across all evaluated models,
including case-by-case paired contingency transitions and McNemar/exact binomial tests.
"""

import os
import json
import glob
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_results")
BASELINE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "01_dual_benchmark_baselines"))
if not os.path.exists(BASELINE_DIR):
    BASELINE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "00_Baseline_Sweep"))
BASELINE_SUMMARY_PATH = os.path.join(BASELINE_DIR, "baseline_summary.json")
BASELINE_RAW_DIR = os.path.join(BASELINE_DIR, "raw_results", "injecagent")

ARCH_MAP = {
    "ornith": ("Ornith-1.5-9B", "9B", "Standard Dense"),
    "qwen9b": ("Qwen3.5-9B", "9B", "Gated DeltaNet Hybrid"),
    "qwen": ("Qwen3.5-4B", "4B", "Gated DeltaNet Hybrid"),
    "gemma": ("Gemma-4-E4B-it", "4B", "Sliding Window Hybrid"),
    "gemma12b": ("Gemma-4-12B-it", "12B", "Sliding Window Hybrid"),
    "nanbeige": ("Nanbeige-4.2-3B", "3B", "Looped Transformer (22x2)"),
    "nano": ("Nemotron-3-Nano-4B", "4B", "Mamba2-Transformer Hybrid"),
    "ministral": ("Ministral-3-14B", "14B", "Dense Reasoning"),
    "lfm": ("LFM-2.5-2.6B", "2.6B", "Liquid SSM Hybrid"),
}

def load_baseline_summary():
    if os.path.exists(BASELINE_SUMMARY_PATH):
        with open(BASELINE_SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("injecagent", {})
    return {}

def analyze_exp2a():
    base_summary = load_baseline_summary()
    # Order models by descending baseline vulnerability
    model_order = ["ornith", "qwen9b", "qwen", "gemma", "gemma12b", "nano", "nanbeige", "ministral", "lfm"]
    models = [m for m in model_order if m in os.listdir(RAW_DIR)]


    results = {}
    print("=" * 115)
    print("TABLE 1: INJECAGENT FULL SWEEP: GUARDRAIL INTERDICTION ACROSS ALL 9 EVALUATED MODELS (N=1,054)")
    print("=" * 115)
    print(f"{'Model':<16} {'Arch':<26} {'Base ASR':>10} {'2A ASR':>10} {'Delta ASR':>11} {'Eta_def':>10} {'Base Inv':>9} {'2A Inv':>8} {'Eval/Tot':>10}")
    print("-" * 115)

    contingency_results = {}
    dh_ds_results = {}

    for m in models:
        m_dir = os.path.join(RAW_DIR, m, "fp16")
        if not os.path.isdir(m_dir):
            continue

        # Load EXP 2A case files directly for authoritative counts
        files = glob.glob(os.path.join(m_dir, "*_case_*.json"))
        e_succ = e_unsucc = e_inval = 0
        dh_e_succ = ds_e_succ = dh_e_tot = ds_e_tot = 0

        e_case_map = {}
        for f in files:
            fn = os.path.basename(f)
            try:
                d = json.load(open(f, encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            ev = d.get("eval")
            e_case_map[fn] = ev

            is_dh = "direct_harm" in fn or "dh" in fn
            is_ds = "data_stealing" in fn or "ds" in fn
            if is_dh:
                dh_e_tot += 1
                if ev == "succ": dh_e_succ += 1
            elif is_ds:
                ds_e_tot += 1
                if ev == "succ": ds_e_succ += 1

            if ev == "succ": e_succ += 1
            elif ev == "unsucc": e_unsucc += 1
            else: e_inval += 1

        tot_cases = 1054
        n_eval = len(files)
        n_fail = tot_cases - n_eval
        e_asr = (e_succ / tot_cases) * 100

        # Baseline stats from 00_Baseline_Sweep
        base_key = f"{m}_fp16"
        base_item = base_summary.get(base_key, {})
        b_succ = base_item.get("succ", 0)
        b_inv = base_item.get("invalid", 0)
        b_asr = base_item.get("asr", round((b_succ / tot_cases) * 100, 2))

        delta_asr = e_asr - b_asr
        if b_asr > 0:
            eta_def = ((b_asr - e_asr) / b_asr) * 100
            eta_str = f"{eta_def:+7.2f}%"
        else:
            eta_def = None
            eta_str = "N/A"

        display_name, param, arch = ARCH_MAP.get(m, (m, "Unknown", "Unknown"))
        print(f"{display_name:<16} {arch:<26} {b_asr:>9.2f}% {e_asr:>9.2f}% {delta_asr:>+10.2f}pp {eta_str:>10} {b_inv:>9d} {e_inval:>8d} {n_eval:>5d}/{tot_cases}")

        # Baseline raw files for paired matched contingency
        b_dir = os.path.join(BASELINE_RAW_DIR, m, "fp16")
        b_files = glob.glob(os.path.join(b_dir, "*_case_*.json")) if os.path.isdir(b_dir) else []
        b_case_map = {}
        dh_b_succ = ds_b_succ = dh_b_tot = ds_b_tot = 0
        for f in b_files:
            fn = os.path.basename(f)
            try:
                d = json.load(open(f, encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            ev = d.get("eval")
            b_case_map[fn] = ev

            is_dh = "direct_harm" in fn or "dh" in fn
            is_ds = "data_stealing" in fn or "ds" in fn
            if is_dh:
                dh_b_tot += 1
                if ev == "succ": dh_b_succ += 1
            elif is_ds:
                ds_b_tot += 1
                if ev == "succ": ds_b_succ += 1

        # Paired contingency table
        common = sorted(list(set(b_case_map.keys()) & set(e_case_map.keys())))
        ss = sf = fs = ff = 0
        for fn in common:
            b_ev = b_case_map[fn]
            e_ev = e_case_map[fn]
            if b_ev == "succ" and e_ev == "succ": ss += 1
            elif b_ev == "succ" and e_ev != "succ": sf += 1
            elif b_ev != "succ" and e_ev == "succ": fs += 1
            else: ff += 1

        net = sf - fs
        n_disc = sf + fs
        if n_disc > 0:
            mcnemar_chi2 = (abs(sf - fs) - 1)**2 / n_disc
            mcnemar_p = stats.chi2.sf(mcnemar_chi2, df=1)
            binom_p = stats.binomtest(sf, n_disc, 0.5, alternative="two-sided").pvalue
        else:
            mcnemar_chi2 = 0.0
            mcnemar_p = 1.0
            binom_p = 1.0

        contingency_results[m] = {
            "model_name": display_name,
            "architecture": arch,
            "n_paired": len(common),
            "ss": ss, "sf": sf, "fs": fs, "ff": ff,
            "net_defensive_flips": net,
            "mcnemar_chi2": round(mcnemar_chi2, 4),
            "mcnemar_p": mcnemar_p,
            "binom_p": binom_p
        }

        dh_ds_results[m] = {
            "model_name": display_name,
            "base_dh_succ": dh_b_succ, "base_dh_tot": dh_b_tot, "base_dh_asr": round(dh_b_succ / dh_b_tot * 100, 2) if dh_b_tot else 0.0,
            "exp_dh_succ": dh_e_succ, "exp_dh_tot": dh_e_tot, "exp_dh_asr": round(dh_e_succ / dh_e_tot * 100, 2) if dh_e_tot else 0.0,
            "base_ds_succ": ds_b_succ, "base_ds_tot": ds_b_tot, "base_ds_asr": round(ds_b_succ / ds_b_tot * 100, 2) if ds_b_tot else 0.0,
            "exp_ds_succ": ds_e_succ, "exp_ds_tot": ds_e_tot, "exp_ds_asr": round(ds_e_succ / ds_e_tot * 100, 2) if ds_e_tot else 0.0,
        }

        results[f"{m}_fp16"] = {
            "model_name": display_name,
            "architecture": arch,
            "total_cases": tot_cases,
            "evaluated": n_eval,
            "failed_or_timeout": n_fail,
            "succ": e_succ,
            "unsucc": e_unsucc,
            "invalid": e_inval,
            "exp2a_asr": round(e_asr, 2),
            "baseline_asr": b_asr,
            "baseline_succ": b_succ,
            "baseline_invalid": b_inv,
            "delta_asr": round(delta_asr, 2),
            "defense_efficiency_pct": round(eta_def, 2) if eta_def is not None else None,
            "dh_breakdown": {
                "succ": dh_e_succ, "total": dh_e_tot, "asr": round(dh_e_succ / dh_e_tot * 100, 2) if dh_e_tot else 0.0
            },
            "ds_breakdown": {
                "succ": ds_e_succ, "total": ds_e_tot, "asr": round(ds_e_succ / ds_e_tot * 100, 2) if ds_e_tot else 0.0
            }
        }

    print("\n" + "=" * 115)
    print("TABLE 2: MATCHED-PAIR CONTINGENCY TRANSITIONS & STATISTICAL SIGNIFICANCE")
    print("=" * 115)
    print(f"{'Model':<16} {'SS':>5} {'SF (Def)':>10} {'FS (Vuln)':>10} {'FF':>6} {'N_pair':>7} {'Net Flips':>10} {'McNemar chi2':>13} {'Binomial p':>14}")
    print("-" * 115)
    for m in models:
        c = contingency_results[m]
        print(f"{c['model_name']:<16} {c['ss']:>5d} {c['sf']:>10d} {c['fs']:>10d} {c['ff']:>6d} {c['n_paired']:>7d} {c['net_defensive_flips']:>+10d} {c['mcnemar_chi2']:>13.2f} {c['binom_p']:>14.2e}")

    print("\n" + "=" * 115)
    print("TABLE 3: DIRECT HARM (DH) VS DATA STEALING (DS) DISSECTION")
    print("=" * 115)
    print(f"{'Model':<16} {'Base DH':>10} {'2A DH':>10} {'DH Delta':>10} {'Base DS':>10} {'2A DS':>10} {'DS Delta':>10}")
    print("-" * 115)
    for m in models:
        d = dh_ds_results[m]
        dh_delta = d['exp_dh_asr'] - d['base_dh_asr']
        ds_delta = d['exp_ds_asr'] - d['base_ds_asr']
        print(f"{d['model_name']:<16} {d['base_dh_asr']:>9.2f}% {d['exp_dh_asr']:>9.2f}% {dh_delta:>+9.2f}pp {d['base_ds_asr']:>9.2f}% {d['exp_ds_asr']:>9.2f}% {ds_delta:>+9.2f}pp")

    output_payload = {
        "metadata": {
            "benchmark": "InjecAgent",
            "total_cases_per_model": 1054,
            "dh_cases": 510,
            "ds_cases": 544,
            "num_models_evaluated": len(models),
            "unexecuted_models": ["minicpm5", "spark"],
            "intervention": "Instruction Source Boundary Guardrails"
        },
        "per_model_results": results,
        "contingency_matrix": contingency_results,
        "dh_ds_dissection": dh_ds_results
    }

    out_path = os.path.join(BASE_DIR, "exp2a_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[+] Full audited results saved to {out_path}")

if __name__ == "__main__":
    analyze_exp2a()
