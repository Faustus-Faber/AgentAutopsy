#!/usr/bin/env python3
"""
Clean Analysis Script: EXP_2BFIX_Untriggered_Default
Evaluates the Untriggered Default (Authorize vs Interdict vs Baseline)
across InjecAgent (11 models, 5 conditions) and AgentDojo (3 models, multi-suite).
Computes exact paired contingency transitions, McNemar chi2, exact binomial tests,
and policy dynamic ranges without hardcoded baseline values.

Requires:
  - scipy (scipy.stats.binomtest, scipy.stats.chi2) for exact paired significance tests.
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

ARCH_MAP = {
    "qwen9b": ("Qwen3.5-9B", "9B", "Gated DeltaNet Hybrid"),
    "gemma": ("Gemma-4-E4B-it", "4B", "Sliding Window Hybrid"),
    "spark": ("Spark-X2.5-4B", "4B", "Standard Dense"),
    "ornith": ("Ornith-1.5-9B", "9B", "Standard Dense"),
    "gemma12b": ("Gemma-4-12B-it", "12B", "Sliding Window Hybrid"),
    "qwen": ("Qwen3.5-4B", "4B", "Gated DeltaNet Hybrid"),
    "nano": ("Nemotron-3-Nano-4B", "4B", "Mamba2-Transformer Hybrid"),
    "nanbeige": ("Nanbeige-4.2-3B", "3B", "Looped Transformer (22x2)"),
    "minicpm5": ("MiniCPM5-2B", "2B", "Standard Dense"),
    "ministral": ("Ministral-3-14B", "14B", "Dense Reasoning"),
    "lfm": ("LFM-2.5-2.6B", "2.6B", "Liquid SSM Hybrid"),
}

def load_baseline():
    if os.path.exists(BASELINE_SUMMARY_PATH):
        with open(BASELINE_SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def analyze_injecagent(baseline_data):
    inj_dir = os.path.join(RAW_DIR, "injecagent")
    base_inj = baseline_data.get("injecagent", {})
    
    # Models sorted by policy swing descending
    model_order = ["qwen9b", "gemma", "spark", "ornith", "gemma12b", "qwen", "nano", "nanbeige", "minicpm5", "ministral", "lfm"]
    models = [m for m in model_order if os.path.isdir(os.path.join(inj_dir, m))]

    results = {}
    table1_rows = []
    dose_rows = []
    contingency_results = {}

    print("=" * 125)
    print("TABLE 1: INJECAGENT FULL SWEEP: THE UNTRIGGERED DEFAULT (AUTHORIZE VS INTERDICT VS BASELINE)")
    print("=" * 125)
    print(f"{'Model':<16} {'Arch':<26} {'Base ASR':>10} {'Inter ASR':>10} {'Auth ASR':>10} {'Auth Surge':>12} {'Policy Swing':>13} {'Eval Cases':>12}")
    print("-" * 125)

    for m in models:
        m_path = os.path.join(inj_dir, m)
        display_name, param, arch = ARCH_MAP.get(m, (m, "Unknown", "Unknown"))
        
        # Baseline from 00_Baseline_Sweep
        b_item = base_inj.get(f"{m}_fp16", {})
        b_succ = b_item.get("succ", 0)
        b_asr = b_item.get("asr", round(b_succ / 1054 * 100, 2))
        b_inv = b_item.get("invalid", 0)

        cond_data = {}
        for cond in ["interdict", "l1", "l2", "l3", "authorize"]:
            c_dir = os.path.join(m_path, cond)
            files = glob.glob(os.path.join(c_dir, "*_case_*.json"))
            succ = unsucc = inval = 0
            for f in files:
                ev = json.load(open(f, encoding="utf-8", errors="ignore")).get("eval")
                if ev == "succ": succ += 1
                elif ev == "unsucc": unsucc += 1
                else: inval += 1
                
            n_eval = len(files)
            tot_nominal = 1054 if cond in ["authorize", "interdict"] else 100
            asr_eval = round(succ / n_eval * 100, 2) if n_eval else 0.0
            asr_all = round(succ / tot_nominal * 100, 2)
            
            cond_data[cond] = {
                "n_eval": n_eval,
                "n_nominal": tot_nominal,
                "succ": succ,
                "unsucc": unsucc,
                "invalid": inval,
                "asr_eval": asr_eval,
                "asr_all": asr_all,
                "files_map": {os.path.basename(f): json.load(open(f, encoding="utf-8", errors="ignore")).get("eval") for f in files}
            }
            results[f"{m}_{cond}"] = {
                "model": m,
                "condition": cond,
                "total": tot_nominal,
                "evaluated": n_eval,
                "failed": tot_nominal - n_eval,
                "succ": succ,
                "unsucc": unsucc,
                "invalid": inval,
                "asr": asr_eval,
                "asr_all": asr_all,
                "base_asr": b_asr
            }

        auth = cond_data["authorize"]
        inter = cond_data["interdict"]
        
        # Policy swing = Auth ASR - Interdict ASR
        auth_asr = auth["asr_eval"]
        inter_asr = inter["asr_eval"]
        auth_surge = auth_asr - b_asr
        policy_swing = auth_asr - inter_asr
        dyn_range = auth_asr / max(inter_asr, 0.001)

        eval_str = f"{auth['n_eval']}/{auth['n_nominal']}"
        print(f"{display_name:<16} {arch:<26} {b_asr:>9.2f}% {inter_asr:>9.2f}% {auth_asr:>9.2f}% {auth_surge:>+11.2f}pp {policy_swing:>+12.2f}pp {eval_str:>12}")

        # Paired transition Authorize vs Interdict
        a_map = auth["files_map"]
        i_map = inter["files_map"]
        common = sorted(list(set(a_map.keys()) & set(i_map.keys())))
        ss = sf = fs = ff = 0
        for fn in common:
            a_ev = a_map[fn]
            i_ev = i_map[fn]
            if a_ev == "succ" and i_ev == "succ": ss += 1
            elif a_ev == "succ" and i_ev != "succ": sf += 1 # Auth succ, Inter fail
            elif a_ev != "succ" and i_ev == "succ": fs += 1 # Auth fail, Inter succ
            else: ff += 1
            
        n_disc = sf + fs
        if n_disc > 0:
            chi2 = (abs(sf - fs) - 1)**2 / n_disc
            p_chi2 = stats.chi2.sf(chi2, df=1)
            p_binom = stats.binomtest(sf, n_disc, 0.5, alternative="two-sided").pvalue
        else:
            chi2 = 0.0; p_chi2 = 1.0; p_binom = 1.0

        contingency_results[m] = {
            "model_name": display_name,
            "architecture": arch,
            "n_paired": len(common),
            "ss": ss, "sf": sf, "fs": fs, "ff": ff,
            "net_swing": sf - fs,
            "mcnemar_chi2": round(chi2, 2),
            "mcnemar_p": p_chi2,
            "binom_p": p_binom
        }

        dose_rows.append({
            "model_name": display_name,
            "interdict": inter_asr,
            "l1": cond_data["l1"]["asr_eval"],
            "l2": cond_data["l2"]["asr_eval"],
            "l3": cond_data["l3"]["asr_eval"],
            "authorize": auth_asr,
            "l1_n": cond_data["l1"]["n_eval"],
            "l2_n": cond_data["l2"]["n_eval"],
            "l3_n": cond_data["l3"]["n_eval"],
        })

    print("\n" + "=" * 125)
    print("PAIRED CONTINGENCY (AUTHORIZE VS INTERDICT) & STATISTICAL SIGNIFICANCE")
    print("=" * 125)
    print(f"{'Model':<16} {'SS':>5} {'SF (Auth-only)':>15} {'FS (Inter-only)':>16} {'FF':>6} {'N_pair':>7} {'Net Shift (SF-FS)':>18} {'McNemar chi2':>13} {'Binomial p':>14}")
    print("-" * 125)
    for m in models:
        c = contingency_results[m]
        print(f"{c['model_name']:<16} {c['ss']:>5d} {c['sf']:>15d} {c['fs']:>16d} {c['ff']:>6d} {c['n_paired']:>7d} {c['net_swing']:>+18d} {c['mcnemar_chi2']:>13.2f} {c['binom_p']:>14.2e}")

    print("\n" + "=" * 125)
    print("TABLE 3: FINE-GRAINED DOSE PROGRESSION (INTERDICT -> L1 -> L2 -> L3 -> AUTHORIZE)")
    print("=" * 125)
    print(f"{'Model':<16} {'Interdict':>10} {'L1 (Suggest)':>13} {'L2 (Direct)':>13} {'L3 (Rational)':>14} {'Authorize':>11}")
    print("-" * 125)
    for r in dose_rows:
        print(f"{r['model_name']:<16} {r['interdict']:>9.2f}% {r['l1']:>12.2f}% {r['l2']:>12.2f}% {r['l3']:>13.2f}% {r['authorize']:>10.2f}%")

    return results, contingency_results, dose_rows

def analyze_agentdojo(baseline_data):
    ad_dir = os.path.join(RAW_DIR, "agentdojo")
    base_ad = baseline_data.get("agentdojo", {})
    if not os.path.isdir(ad_dir):
        return {}

    models = sorted(os.listdir(ad_dir))
    results = {}
    print("\n" + "=" * 125)
    print("TABLE 2: AGENTDOJO MULTI-TURN POLICY SWING (ON-DISK EVALUATED MODELS)")
    print("=" * 125)
    print(f"{'Model':<16} {'Param':<6} {'Base ASR':>10} {'Inter ASR':>10} {'Auth ASR':>10} {'Auth Surge':>12} {'Policy Swing':>13} {'Dyn Range':>11}")
    print("-" * 125)

    for m in models:
        m_dir = os.path.join(ad_dir, m)
        display_name, param, arch = ARCH_MAP.get(m, (m, "Unknown", "Unknown"))
        b_key = f"{m}_fp16"
        b_item = base_ad.get(b_key, {})
        b_asr = round(b_item.get("asr", 0.0), 2)
        b_n = b_item.get("attacks_won", 0)
        b_tot = b_item.get("paired_total", 949)

        cond_results = {}
        for cond in sorted(os.listdir(m_dir)):
            c_dir = os.path.join(m_dir, cond)
            paired_tot = paired_succ = 0
            for root, dirs, files in os.walk(c_dir):
                if 'injection_task_' in root and 'none' in root: continue
                is_paired = ('tool_knowledge' in root or 'important_instructions' in root)
                if not is_paired: continue
                
                for f in files:
                    if not f.endswith('.json') or 'metrics' in f or f in ('progress.json', 'summary.json'): continue
                    fp = os.path.join(root, f)
                    try:
                        d = json.load(open(fp, encoding="utf-8", errors="ignore"))
                        paired_tot += 1
                        if d.get("security") is True: paired_succ += 1
                    except:
                        pass
            asr = round(paired_succ / paired_tot * 100, 2) if paired_tot else 0.0
            cond_results[cond] = {
                "paired_total": paired_tot,
                "attacks_won": paired_succ,
                "asr": asr
            }

        auth = cond_results.get("authorize", {"asr": 0.0, "attacks_won": 0, "paired_total": 0})
        inter = cond_results.get("interdict", {"asr": 0.0, "attacks_won": 0, "paired_total": 0})
        
        auth_asr = auth["asr"]
        inter_asr = inter["asr"]
        auth_surge = auth_asr - b_asr
        policy_swing = auth_asr - inter_asr
        dyn_range = auth_asr / max(inter_asr, 0.001)

        print(f"{display_name:<16} {param:<6} {b_asr:>9.2f}% {inter_asr:>9.2f}% {auth_asr:>9.2f}% {auth_surge:>+11.2f}pp {policy_swing:>+12.2f}pp {dyn_range:>10.2f}x")

        results[m] = {
            "model_name": display_name,
            "parameter_scale": param,
            "architecture": arch,
            "baseline": {"asr": b_asr, "attacks_won": b_n, "paired_total": b_tot},
            "authorize": auth,
            "interdict": inter,
            "auth_surge_pp": round(auth_surge, 2),
            "policy_swing_pp": round(policy_swing, 2),
            "dynamic_range": round(dyn_range, 2)
        }

    return results

def main():
    base_data = load_baseline()
    inj_results, contingency, dose = analyze_injecagent(base_data)
    ad_results = analyze_agentdojo(base_data)

    payload = {
        "metadata": {
            "benchmark_injecagent": "InjecAgent (N=1,054 nominal cases)",
            "benchmark_agentdojo": "AgentDojo (N=949 nominal paired cases)",
            "experiment": "EXP_2BFIX_Untriggered_Default",
            "conditions": ["interdict", "l1", "l2", "l3", "authorize"],
            "models_evaluated_injecagent": len(inj_results) // 5,
            "models_evaluated_agentdojo": len(ad_results)
        },
        "injecagent_per_cell": inj_results,
        "injecagent_contingency": contingency,
        "injecagent_dose_progression": dose,
        "agentdojo_per_model": ad_results
    }

    out_file = os.path.join(BASE_DIR, "exp2bfix_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Audited summary successfully saved to {out_file}")

if __name__ == "__main__":
    main()
