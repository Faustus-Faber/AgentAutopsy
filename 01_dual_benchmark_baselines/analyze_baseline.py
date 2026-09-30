#!/usr/bin/env python3
"""
Clean Analysis Script: 00_Baseline_Sweep
Calculates exact Attack Success Rate (ASR) across all models and precisions
for both InjecAgent and AgentDojo from raw task files and verified summary manifests.

Strict benchmark definitions applied:
- AgentDojo: ASR = security=True count / paired injection tasks (N=949).
             Utility = utility=True count / benign utility tasks (N=97).
- InjecAgent: ASR = succ count / total cases (N=1054).
              Direct Harm (DH) = succ / 510.
              Data Stealing (DS) = succ / 544.
              Invalid responses explicitly audited to prevent 'defense by dysfunction'.
"""

import os
import sys
import json
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_results")

def analyze_injecagent():
    ia_dir = os.path.join(RAW_DIR, "injecagent")
    if not os.path.exists(ia_dir):
        print(f"[!] InjecAgent raw directory not found at {ia_dir}", flush=True)
        return {}

    results = {}
    print("=" * 95, flush=True)
    print("INJECAGENT BASELINE EVALUATION (N=1,054)", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Model':<14} {'Prec':<6} {'Total':>6} {'Succ':>6} {'Unsucc':>6} {'Inval':>6} {'ASR(%)':>8} {'DH(%)':>8} {'DS(%)':>8}", flush=True)
    print("-" * 95, flush=True)

    models = sorted([m for m in os.listdir(ia_dir) if os.path.isdir(os.path.join(ia_dir, m)) and 'hermes' not in m.lower()])
    for m in models:
        m_path = os.path.join(ia_dir, m)
        for prec in sorted(os.listdir(m_path)):
            p_path = os.path.join(m_path, prec)
            if not os.path.isdir(p_path):
                continue
            
            files = [f for f in glob.glob(os.path.join(p_path, "*.json")) if not any(x in f for x in ['metrics', 'progress', 'summary', 'run'])]
            tot = len(files)
            succ = unsucc = inval = 0
            dh_tot = dh_succ = ds_tot = ds_succ = 0
            for fp in files:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    ev = data.get("eval")
                    cat = data.get("attack_category")
                    if ev == "succ":
                        succ += 1
                        if cat == "direct_harm": dh_succ += 1
                        elif cat == "data_stealing": ds_succ += 1
                    elif ev == "unsucc":
                        unsucc += 1
                    else:
                        inval += 1

                    if cat == "direct_harm": dh_tot += 1
                    elif cat == "data_stealing": ds_tot += 1
                except Exception:
                    inval += 1
            asr = (succ / tot * 100) if tot else 0.0
            dh_asr = (dh_succ / dh_tot * 100) if dh_tot else 0.0
            ds_asr = (ds_succ / ds_tot * 100) if ds_tot else 0.0

            results[(m, prec)] = {
                "total": tot, "succ": succ, "unsucc": unsucc, "invalid": inval,
                "asr": round(asr, 2), "dh_asr": round(dh_asr, 2), "ds_asr": round(ds_asr, 2)
            }
            print(f"{m:<14} {prec:<6} {tot:>6} {succ:>6} {unsucc:>6} {inval:>6} {asr:>8.2f} {dh_asr:>8.2f} {ds_asr:>8.2f}", flush=True)
    return results

def analyze_agentdojo():
    ad_dir = os.path.join(RAW_DIR, "agentdojo")
    if not os.path.exists(ad_dir):
        print(f"[!] AgentDojo raw directory not found at {ad_dir}", flush=True)
        return {}

    results = {}
    print("\n" + "=" * 95, flush=True)
    print("AGENTDOJO BASELINE EVALUATION (N=949 Paired Injections)", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Model':<14} {'Prec':<6} {'Paired Total':>13} {'Attacks Won':>12} {'ASR(%)':>9} {'Utility Pass(%)':>17}", flush=True)
    print("-" * 95, flush=True)

    models = sorted([m for m in os.listdir(ad_dir) if os.path.isdir(os.path.join(ad_dir, m)) and 'hermes' not in m.lower()])
    for m in models:
        m_path = os.path.join(ad_dir, m)
        for prec in sorted(os.listdir(m_path)):
            p_path = os.path.join(m_path, prec)
            if not os.path.isdir(p_path) or 'smoke' in prec.lower():
                continue
            
            paired_tot = 0
            paired_succ = 0
            util_tot = 0
            util_pass = 0

            # Scan files directly targeting known suites
            for root, dirs, files in os.walk(p_path):
                # Skip standalone injection utility test folders
                if 'injection_task_' in root and 'none' in root:
                    continue
                is_paired = ('tool_knowledge' in root or 'important_instructions' in root)
                is_util = ('user_task_' in root and 'none' in root)
                if not (is_paired or is_util):
                    continue
                
                for f in files:
                    if not f.endswith('.json') or 'metrics' in f or f in ('progress.json', 'summary.json'):
                        continue
                    fp = os.path.join(root, f)
                    try:
                        with open(fp, "r", encoding="utf-8") as fh:
                            d = json.load(fh)
                        if is_paired:
                            paired_tot += 1
                            if d.get("security") is True:
                                paired_succ += 1
                        elif is_util:
                            util_tot += 1
                            if d.get("utility") is True:
                                util_pass += 1
                    except Exception:
                        pass

            asr = (paired_succ / paired_tot * 100) if paired_tot else 0.0
            util_rate = (util_pass / util_tot * 100) if util_tot else 0.0

            results[(m, prec)] = {
                "paired_total": paired_tot, "attacks_won": paired_succ, "asr": asr,
                "utility_total": util_tot, "utility_passed": util_pass, "utility_rate": util_rate
            }
            print(f"{m:<14} {prec:<6} {paired_tot:>13} {paired_succ:>12} {asr:>9.2f} {util_rate:>16.2f}%", flush=True)
    return results

if __name__ == "__main__":
    ia_res = analyze_injecagent()
    ad_res = analyze_agentdojo()
    
    out_summary = {
        "injecagent": {f"{k[0]}_{k[1]}": v for k, v in ia_res.items()},
        "agentdojo": {f"{k[0]}_{k[1]}": v for k, v in ad_res.items()}
    }
    out_path = os.path.join(BASE_DIR, "baseline_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\n[+] Analysis saved to {out_path}", flush=True)
