#!/usr/bin/env python3
"""
Audited Analysis Script: EXP_3A_Context_Invariance
Evaluates whether context completion budget constraints (16k, 8k, 4k, 2k, 1k, 512, 256 tokens)
affect prompt injection attack success rate (ASR).

Authoritative metrics are parsed directly from individual case JSON files (eval == 'succ'),
bypassing stale per-cell summary.json files.
"""

import os
import json
import glob
import math
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_results")
BASELINE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "01_dual_benchmark_baselines", "raw_results", "injecagent", "nanbeige", "fp16"))
if not os.path.exists(BASELINE_DIR):
    BASELINE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "00_Baseline_Sweep", "raw_results", "injecagent", "nanbeige", "fp16"))

def count_case_files(directory):
    """Parse case JSON files directly to determine authoritative evaluation counts."""
    files = [f for f in glob.glob(os.path.join(directory, "*.json"))
             if not any(x in os.path.basename(f) for x in ("metrics", "progress", "summary", "run"))]
    total = len(files)
    succ = unsucc = invalid = 0
    dh_total = dh_succ = ds_total = ds_succ = 0
    reasoning_lengths = []
    case_map = {}

    for fp in files:
        bn = os.path.basename(fp)
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                j = json.load(f)
        except Exception:
            j = {}
        ev = j.get("eval")
        case_map[bn] = ev
        if ev == "succ":
            succ += 1
        elif ev == "unsucc":
            unsucc += 1
        else:
            invalid += 1

        is_dh = "direct_harm" in bn
        is_ds = "data_stealing" in bn
        if is_dh:
            dh_total += 1
            if ev == "succ":
                dh_succ += 1
        elif is_ds:
            ds_total += 1
            if ev == "succ":
                ds_succ += 1

        t1 = j.get("thinking") or ""
        t2 = j.get("thinking_step2") or ""
        t_len = len(t1) + (len(t2) if isinstance(t2, str) else 0)
        reasoning_lengths.append(t_len)

    asr_all = round(succ / total * 100, 2) if total else 0.0
    valid_count = total - invalid
    asr_valid = round(succ / valid_count * 100, 2) if valid_count else 0.0
    inv_rate = round(invalid / total * 100, 2) if total else 0.0
    dh_asr = round(dh_succ / dh_total * 100, 2) if dh_total else 0.0
    ds_asr = round(ds_succ / ds_total * 100, 2) if ds_total else 0.0
    avg_chars = round(sum(reasoning_lengths) / len(reasoning_lengths)) if reasoning_lengths else 0
    max_chars = max(reasoning_lengths) if reasoning_lengths else 0

    return {
        "total": total,
        "succ": succ,
        "unsucc": unsucc,
        "invalid": invalid,
        "asr_all": asr_all,
        "asr_valid": asr_valid,
        "invalid_rate": inv_rate,
        "dh_total": dh_total,
        "dh_succ": dh_succ,
        "dh_asr": dh_asr,
        "ds_total": ds_total,
        "ds_succ": ds_succ,
        "ds_asr": ds_asr,
        "avg_reasoning_chars": avg_chars,
        "max_reasoning_chars": max_chars,
        "case_map": case_map,
    }

def exact_binom_p(k, n, p0=0.5):
    """Two-sided exact binomial test."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i)) for i in range(n + 1)]
    pk = probs[k]
    return min(1.0, sum(p for p in probs if p <= pk + 1e-15))

def mcnemar_chi2_p(sf, fs):
    """McNemar chi-square test with Edwards continuity correction."""
    n_disc = sf + fs
    if n_disc == 0:
        return 0.0, 1.0
    chi2 = (abs(sf - fs) - 1) ** 2 / n_disc
    p_val = math.erfc(math.sqrt(chi2 / 2))
    return round(chi2, 4), p_val

def analyze_exp3a():
    print("=" * 100)
    print("EXP 3A AUDITED ANALYSIS: CONTEXT AND REASONING BUDGET INVARIANCE")
    print("=" * 100)

    # 1. Macro Sweep (Table 1)
    macro_models = [
        ("qwen", "Qwen3.5-4B", "Gated DeltaNet Hybrid"),
        ("qwen9b", "Qwen3.5-9B", "Gated DeltaNet Hybrid"),
        ("gemma", "Gemma-4-E4B-it", "Sliding Window Hybrid"),
        ("gemma12b", "Gemma-4-12B-it", "Sliding Window Hybrid"),
        ("nano", "Nemotron-3-Nano-4B", "Mamba2-Transformer Hybrid"),
        ("ministral", "Ministral-3-14B", "Dense Reasoning"),
        ("ornith", "Ornith-1.5-9B", "Standard Dense"),
        ("minicpm5", "MiniCPM5-2B", "Standard Dense"),
        ("spark", "Spark-X2.5-4B", "Standard Dense"),
    ]
    budgets = ["tokens_16000", "tokens_8000", "tokens_4000"]

    print("\n--- TABLE 1: Stage 1 Macro-Budget Invariance (16k / 8k / 4k Tokens) ---")
    print(f"{'Model':<16} {'Architecture':<26} {'16k Tokens (succ/N)':<22} {'8k Tokens (succ/N)':<22} {'4k Tokens (succ/N)':<22} {'Variance':<12}")
    print("-" * 125)

    macro_results = {}
    summary_export = {}

    for mid, mname, arch in macro_models:
        row_data = {"display_name": mname, "architecture": arch, "budgets": {}}
        asrs = []
        for bud in budgets:
            d = os.path.join(RAW_DIR, mid, bud)
            c = count_case_files(d)
            row_data["budgets"][bud] = {
                "total": c["total"],
                "succ": c["succ"],
                "unsucc": c["unsucc"],
                "invalid": c["invalid"],
                "asr_all": c["asr_all"],
                "asr_valid": c["asr_valid"],
            }
            summary_export[f"{mid}_{bud}"] = {
                "model": mid,
                "budget": bud,
                "total": c["total"],
                "succ": c["succ"],
                "unsucc": c["unsucc"],
                "invalid": c["invalid"],
                "asr": c["asr_all"],
                "asr_valid": c["asr_valid"],
            }
            asrs.append(c["asr_all"])

        spread = round(max(asrs) - min(asrs), 2)
        row_data["variance_pp"] = spread
        macro_results[mid] = row_data

        b16 = f"{row_data['budgets']['tokens_16000']['asr_all']:.2f}% ({row_data['budgets']['tokens_16000']['succ']}/{row_data['budgets']['tokens_16000']['total']})"
        b8 = f"{row_data['budgets']['tokens_8000']['asr_all']:.2f}% ({row_data['budgets']['tokens_8000']['succ']}/{row_data['budgets']['tokens_8000']['total']})"
        b4 = f"{row_data['budgets']['tokens_4000']['asr_all']:.2f}% ({row_data['budgets']['tokens_4000']['succ']}/{row_data['budgets']['tokens_4000']['total']})"
        print(f"{mname:<16} {arch:<26} {b16:<22} {b8:<22} {b4:<22} {spread:.2f} pp")

    # 2. Nanbeige Deep Sweep (Table 2)
    print("\n--- TABLE 2: Stage 2 Deep Micro-Budget Dose-Response (Nanbeige-4.2-3B, N=1,054) ---")
    print(f"{'Budget':<15} {'ASR (All)':<16} {'ASR (Valid)':<14} {'Invalid Rate':<18} {'Avg Chars':<12} {'Max Chars':<12} {'DH ASR':<14} {'DS ASR':<14}")
    print("-" * 115)

    deep_budgets = ["baseline", "tokens_2000", "tokens_1000", "tokens_512", "tokens_256"]
    deep_results = {}
    baseline_case_map = {}

    for bud in deep_budgets:
        if bud == "baseline":
            d = BASELINE_DIR
            c = count_case_files(d)
            baseline_case_map = c["case_map"]
            summary_export["nanbeige_baseline"] = {
                "model": "nanbeige",
                "budget": "baseline",
                "total": c["total"],
                "succ": c["succ"],
                "unsucc": c["unsucc"],
                "invalid": c["invalid"],
                "asr": c["asr_all"],
                "asr_valid": c["asr_valid"],
            }
        else:
            d = os.path.join(RAW_DIR, "nanbeige", bud)
            c = count_case_files(d)
            summary_export[f"nanbeige_{bud}"] = {
                "model": "nanbeige",
                "budget": bud,
                "total": c["total"],
                "succ": c["succ"],
                "unsucc": c["unsucc"],
                "invalid": c["invalid"],
                "asr": c["asr_all"],
                "asr_valid": c["asr_valid"],
            }

        deep_results[bud] = {
            "total": c["total"],
            "succ": c["succ"],
            "unsucc": c["unsucc"],
            "invalid": c["invalid"],
            "asr_all": c["asr_all"],
            "asr_valid": c["asr_valid"],
            "invalid_rate": c["invalid_rate"],
            "avg_reasoning_chars": c["avg_reasoning_chars"],
            "max_reasoning_chars": c["max_reasoning_chars"],
            "dh_total": c["dh_total"],
            "dh_succ": c["dh_succ"],
            "dh_asr": c["dh_asr"],
            "ds_total": c["ds_total"],
            "ds_succ": c["ds_succ"],
            "ds_asr": c["ds_asr"],
            "case_map": c["case_map"],
        }

        asr_all_str = f"{c['asr_all']:.2f}% ({c['succ']})"
        asr_val_str = f"{c['asr_valid']:.2f}%"
        inv_str = f"{c['invalid_rate']:.2f}% ({c['invalid']})"
        dh_str = f"{c['dh_asr']:.2f}% ({c['dh_succ']})"
        ds_str = f"{c['ds_asr']:.2f}% ({c['ds_succ']})"
        print(f"{bud:<15} {asr_all_str:<16} {asr_val_str:<14} {inv_str:<18} {c['avg_reasoning_chars']:<12} {c['max_reasoning_chars']:<12} {dh_str:<14} {ds_str:<14}")

    # 3. Matched-Pair Contingency Table (Table 3)
    print("\n--- TABLE 3: Matched-Pair Contingency Transitions vs. Baseline (N=1,054) ---")
    print(f"{'Comparison':<28} {'SS':<6} {'SF (Defended)':<15} {'FS (Backfire)':<15} {'FF':<8} {'Net':<8} {'Binom p':<14} {'McNemar chi2':<14} {'McNemar p':<12}")
    print("-" * 125)

    contingency_results = {}
    for bud in ["tokens_2000", "tokens_1000", "tokens_512", "tokens_256"]:
        bmap = baseline_case_map
        emap = deep_results[bud]["case_map"]
        common = sorted(set(bmap.keys()) & set(emap.keys()))
        ss = sf = fs = ff = 0
        for k in common:
            b_val = bmap[k]
            e_val = emap[k]
            if b_val == "succ" and e_val == "succ":
                ss += 1
            elif b_val == "succ" and e_val != "succ":
                sf += 1
            elif b_val != "succ" and e_val == "succ":
                fs += 1
            else:
                ff += 1

        net = sf - fs
        bp = exact_binom_p(sf, sf + fs)
        chi2, mp = mcnemar_chi2_p(sf, fs)

        contingency_results[bud] = {
            "n_paired": len(common),
            "ss": ss,
            "sf_defended": sf,
            "fs_backfire": fs,
            "ff": ff,
            "net_flipped": net,
            "exact_binomial_p": bp,
            "mcnemar_chi2": chi2,
            "mcnemar_p": mp,
        }
        comp_name = f"Baseline vs. {bud.replace('tokens_', '')} Tokens"
        print(f"{comp_name:<28} {ss:<6} {sf:<15} {fs:<15} {ff:<8} {net:<+8} {bp:<14.6e} {chi2:<14.4f} {mp:<12.6e}")

    # 4. Utility-Security Tradeoff Table (Table 4)
    print("\n--- TABLE 4: The Severe Utility-Security Tradeoff ---")
    print(f"{'Budget':<15} {'Valid Outputs':<16} {'ASR (Valid)':<14} {'Security Gain':<16} {'Utility Cost':<16} {'Tradeoff Ratio':<15}")
    print("-" * 95)

    base_inv_rate = deep_results["baseline"]["invalid_rate"]
    base_asr_all = deep_results["baseline"]["asr_all"]
    tradeoff_results = {}

    for bud in deep_budgets:
        inv_r = deep_results[bud]["invalid_rate"]
        valid_out = round(100.0 - inv_r, 2)
        asr_val = deep_results[bud]["asr_valid"]
        if bud == "baseline":
            sec_gain = 0.0
            util_cost = 0.0
            ratio_str = "—"
            ratio_val = None
        else:
            sec_gain = round(base_asr_all - deep_results[bud]["asr_all"], 2)
            util_cost = round(inv_r - base_inv_rate, 2)
            ratio_val = round(util_cost / sec_gain, 1) if sec_gain > 0 else 0.0
            ratio_str = f"1 : {ratio_val}"

        tradeoff_results[bud] = {
            "valid_outputs_pct": valid_out,
            "asr_valid_pct": asr_val,
            "security_gain_pp": sec_gain,
            "utility_cost_pp": util_cost,
            "tradeoff_ratio": ratio_val,
        }
        bud_label = bud.replace("tokens_", "")
        vo_str = f"{valid_out:.2f}%"
        av_str = f"{asr_val:.2f}%"
        sg_str = f"{sec_gain:+.2f} pp"
        uc_str = f"{util_cost:+.2f} pp"
        print(f"{bud_label:<15} {vo_str:<16} {av_str:<14} {sg_str:<16} {uc_str:<16} {ratio_str:<15}")

    # Clean case_map from deep_results before export to keep JSON manageable
    for b in deep_results:
        deep_results[b].pop("case_map", None)

    # 5. Export comprehensive summary
    full_output = {
        "macro_sweep": macro_results,
        "nanbeige_deep_sweep": deep_results,
        "matched_pair_contingency": contingency_results,
        "utility_security_tradeoff": tradeoff_results,
        "cell_metrics": summary_export,
    }
    # Also merge cell_metrics at top level for backward compatibility
    for k, v in summary_export.items():
        full_output[k] = v

    out_path = os.path.join(BASE_DIR, "exp3a_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[+] Successfully exported complete audited metrics to: {out_path}")

if __name__ == "__main__":
    analyze_exp3a()
