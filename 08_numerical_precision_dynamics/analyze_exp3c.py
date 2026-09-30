#!/usr/bin/env python3
"""
Audited Analysis Script: EXP_3C_Precision_Dysfunction
Analyzes cross-precision reasoning dynamics and syntax dysfunction across FP16, FP8, and NF4:
- Table 1: Cross-Precision ASR, Category Breakdown (DH vs DS), and Syntax Invalidity Rates
- Table 2: Average Reasoning Lengths, S/F Engagement Ratios, and Wilcoxon Signed-Rank Tests
- Table 3: Bidirectional Matched-Pair Contingency Transitions (FP16 vs NF4) and McNemar Tests

Authoritative data is parsed directly from 00_Baseline_Sweep/raw_results/injecagent case files.
"""

import os
import sys
import json
import glob
import math
from collections import defaultdict
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = os.path.abspath(os.path.join(BASE_DIR, "..", "01_dual_benchmark_baselines", "raw_results", "injecagent"))
if not os.path.exists(DEFAULT_BASE):
    DEFAULT_BASE = os.path.abspath(os.path.join(BASE_DIR, "..", "00_Baseline_Sweep", "raw_results", "injecagent"))
BASELINE_BASE = DEFAULT_BASE if os.path.isdir(DEFAULT_BASE) else "/home/user1/lyceum/thesis/results/injecagent"

MODELS = [
    ("qwen", "Qwen3.5-4B", "4B", "Gated DeltaNet Hybrid"),
    ("gemma", "Gemma-4-E4B-it", "4B", "Sliding Window Hybrid"),
    ("nano", "Nemotron-3-Nano-4B", "4B", "Mamba2-Transformer Hybrid"),
    ("nanbeige", "Nanbeige-4.2-3B", "3B", "Looped Transformer (22×2)"),
    ("lfm", "LFM-2.5-2.6B", "2.6B", "Liquid SSM Hybrid"),
]
PRECISIONS = ["fp16", "fp8", "nf4"]

def load_cell_data(model, precision):
    """Load and parse case JSON files for a specific model and precision."""
    d = os.path.join(BASELINE_BASE, model, precision)
    if not os.path.isdir(d):
        return None

    files = [f for f in sorted(glob.glob(os.path.join(d, "*.json")))
             if not any(x in os.path.basename(f) for x in ("summary", "metrics", "progress"))]
    total = len(files)
    succ = unsucc = invalid = 0
    dh_total = dh_succ = ds_total = ds_succ = 0
    reasoning_lengths = []
    succ_rls = []
    fail_rls = []
    cases = {}

    for fp in files:
        bn = os.path.basename(fp)
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                j = json.load(f)
        except Exception:
            j = {}

        task_id = j.get("user_task_id", bn.replace(".json", ""))
        ev = j.get("eval", "")
        cat = j.get("attack_category", "")
        is_succ = (ev == "succ")

        t1 = j.get("thinking") or ""
        t2 = j.get("thinking_step2") or ""
        t_len = len(t1) + (len(t2) if isinstance(t2, str) else 0)
        reasoning_lengths.append(t_len)

        if is_succ:
            succ += 1
            succ_rls.append(t_len)
        elif ev == "unsucc":
            unsucc += 1
            fail_rls.append(t_len)
        else:
            invalid += 1

        is_dh = ("direct_harm" in bn) or (cat == "direct_harm")
        is_ds = ("data_stealing" in bn) or (cat == "data_stealing")
        if is_dh:
            dh_total += 1
            if is_succ:
                dh_succ += 1
        elif is_ds:
            ds_total += 1
            if is_succ:
                ds_succ += 1

        cases[task_id] = {
            "succ": is_succ,
            "eval": ev,
            "rl": t_len,
            "category": "direct_harm" if is_dh else "data_stealing",
        }

    asr = round(succ / total * 100, 2) if total else 0.0
    valid_count = total - invalid
    valid_rate = round(valid_count / total * 100, 2) if total else 0.0
    dh_asr = round(dh_succ / dh_total * 100, 2) if dh_total else 0.0
    ds_asr = round(ds_succ / ds_total * 100, 2) if ds_total else 0.0
    avg_rl = round(sum(reasoning_lengths) / len(reasoning_lengths)) if reasoning_lengths else 0
    max_rl = max(reasoning_lengths) if reasoning_lengths else 0

    succ_mean_rl = (sum(succ_rls) / len(succ_rls)) if succ_rls else 0.0
    fail_mean_rl = (sum(fail_rls) / len(fail_rls)) if fail_rls else 0.0
    sf_ratio = round(succ_mean_rl / fail_mean_rl, 2) if fail_mean_rl > 0 and succ_mean_rl > 0 else 0.0

    return {
        "total": total,
        "succ": succ,
        "unsucc": unsucc,
        "invalid": invalid,
        "valid_rate": valid_rate,
        "asr": asr,
        "dh_total": dh_total,
        "dh_succ": dh_succ,
        "dh_asr": dh_asr,
        "ds_total": ds_total,
        "ds_succ": ds_succ,
        "ds_asr": ds_asr,
        "avg_rl": avg_rl,
        "max_rl": max_rl,
        "succ_mean_rl": round(succ_mean_rl, 1),
        "fail_mean_rl": round(fail_mean_rl, 1),
        "sf_ratio": sf_ratio,
        "cases": cases,
    }

def mcnemar_edwards(sf, fs):
    """Compute McNemar chi-square test with Edwards continuity correction."""
    n_disc = sf + fs
    if n_disc == 0:
        return 0.0, 1.0
    chi2 = (abs(sf - fs) - 1.0) ** 2 / n_disc
    p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return round(chi2, 4), p_val

def exact_binom_p(k, n, p0=0.5):
    """Two-sided exact binomial test on discordant pairs."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i)) for i in range(n + 1)]
    pk = probs[k]
    return min(1.0, sum(p for p in probs if p <= pk + 1e-15))

def analyze_exp3c():
    print("=" * 115)
    print("EXP 3C AUDITED ANALYSIS: CROSS-PRECISION REASONING DYNAMICS & SYNTAX DYSFUNCTION")
    print("=" * 115)

    all_data = {}
    table1_data = []
    cell_metrics = {}

    for mid, mname, params, arch in MODELS:
        all_data[mid] = {}
        for prec in PRECISIONS:
            cd = load_cell_data(mid, prec)
            if cd is None:
                continue
            all_data[mid][prec] = cd

            row = {
                "model_id": mid,
                "display_name": mname,
                "params": params,
                "architecture": arch,
                "precision": prec.upper(),
                "attacks_won": cd["succ"],
                "total_cases": cd["total"],
                "asr": cd["asr"],
                "dh_asr": cd["dh_asr"],
                "ds_asr": cd["ds_asr"],
                "invalid_calls": cd["invalid"],
                "valid_output_rate": cd["valid_rate"],
            }
            table1_data.append(row)
            cell_metrics[f"{mid}_{prec}"] = {
                "name": mname,
                "prec": prec,
                "asr": cd["asr"],
                "avg_reasoning_len": cd["avg_rl"],
                "sf_ratio": cd["sf_ratio"],
                "invalid_calls": cd["invalid"],
                "valid_output_rate": cd["valid_rate"],
                "dh_asr": cd["dh_asr"],
                "ds_asr": cd["ds_asr"],
            }

    # 1. Print Table 1
    print("\n--- TABLE 1: Cross-Precision ASR and Tool Formatting Validity (N=1,054) ---")
    print(f"{'Model Name':<20} {'Architecture':<26} {'Prec':<5} {'Won':>5} {'ASR (%)':>8} {'DH ASR':>8} {'DS ASR':>8} {'Invalid':>8} {'Valid Rate':>11}")
    print("-" * 115)
    for r in table1_data:
        print(f"{r['display_name']:<20} {r['architecture']:<26} {r['precision']:<5} {r['attacks_won']:>5} {r['asr']:>7.2f}% {r['dh_asr']:>7.2f}% {r['ds_asr']:>7.2f}% {r['invalid_calls']:>8} {r['valid_output_rate']:>10.2f}%")

    # 2. Table 2: Reasoning Length and S/F Ratio
    print("\n--- TABLE 2: Average Reasoning Length (Chars) and S/F Engagement Ratio ---")
    print(f"{'Model Name':<20} {'FP16 RL':>8} {'FP8 RL':>8} {'NF4 RL':>8} {'FP16 S/F':>9} {'FP8 S/F':>9} {'NF4 S/F':>9} {'Wilcoxon p (RL)':>16} {'Engagement Trend':<32}")
    print("-" * 130)

    table2_data = []
    for mid, mname, params, arch in MODELS:
        d16 = all_data[mid].get("fp16")
        d8 = all_data[mid].get("fp8")
        dnf4 = all_data[mid].get("nf4")

        rl16 = d16["avg_rl"] if d16 else 0
        rl8 = d8["avg_rl"] if d8 else 0
        rlnf4 = dnf4["avg_rl"] if dnf4 else 0

        sf16 = d16["sf_ratio"] if d16 else None
        sf8 = d8["sf_ratio"] if d8 else None
        sfnf4 = dnf4["sf_ratio"] if dnf4 else None

        # Paired Wilcoxon signed-rank test on FP16 vs NF4 reasoning lengths
        common_keys = sorted(set(d16["cases"].keys()) & set(dnf4["cases"].keys()))
        pairs_16 = [d16["cases"][k]["rl"] for k in common_keys]
        pairs_nf4 = [dnf4["cases"][k]["rl"] for k in common_keys]
        diffs = [b - a for a, b in zip(pairs_16, pairs_nf4)]
        nonzero_diffs = [x for x in diffs if x != 0]

        if len(nonzero_diffs) > 0:
            w_res = stats.wilcoxon(pairs_16, pairs_nf4)
            w_stat, w_p = round(w_res.statistic, 1), w_res.pvalue
        else:
            w_stat, w_p = 0.0, 1.0

        # Engagement Trend determination (1.15x rule)
        if sf16 is not None and sfnf4 is not None and sf16 > 0:
            if sfnf4 > sf16 * 1.15:
                trend = "NF4 amplifies engagement"
            elif sfnf4 < sf16 * 0.85:
                trend = "NF4 suppresses engagement"
            else:
                trend = "Engagement remains constant"
        else:
            trend = "Syntax collapsed / Insufficient data"

        sf16_str = f"{sf16:.2f}x" if sf16 is not None else "N/A"
        sf8_str = f"{sf8:.2f}x" if sf8 is not None else "N/A"
        sfnf4_str = f"{sfnf4:.2f}x" if sfnf4 is not None else "N/A"
        wp_str = f"p = {w_p:.2e}" if w_p < 0.001 else f"p = {w_p:.4f}"

        print(f"{mname:<20} {rl16:>8} {rl8:>8} {rlnf4:>8} {sf16_str:>9} {sf8_str:>9} {sfnf4_str:>9} {wp_str:>16} {trend:<32}")

        table2_data.append({
            "model_id": mid,
            "display_name": mname,
            "fp16_avg_rl": rl16,
            "fp8_avg_rl": rl8,
            "nf4_avg_rl": rlnf4,
            "fp16_sf_ratio": sf16,
            "fp8_sf_ratio": sf8,
            "nf4_sf_ratio": sfnf4,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "trend": trend,
        })

    # 3. Table 3: Matched-Pair Contingency: FP16 vs NF4
    print("\n--- TABLE 3: Bidirectional Matched-Pair Contingency: FP16 vs. NF4 (N=1,054) ---")
    print(f"{'Model Name':<20} {'S→S':>5} {'S→F (Safer)':>12} {'F→S (Worse)':>12} {'F→F':>6} {'Net Shift (FS-SF)':>18} {'McNemar chi2':>14} {'McNemar p':>14} {'Exact Binom p':>14}")
    print("-" * 125)

    table3_data = []
    for mid, mname, params, arch in MODELS:
        d16 = all_data[mid]["fp16"]["cases"]
        dnf4 = all_data[mid]["nf4"]["cases"]
        common = sorted(set(d16.keys()) & set(dnf4.keys()))

        ss = sf = fs = ff = 0
        for k in common:
            x, y = d16[k]["succ"], dnf4[k]["succ"]
            if x and y:
                ss += 1
            elif x and not y:
                sf += 1
            elif not x and y:
                fs += 1
            else:
                ff += 1

        net_shift = fs - sf
        chi2, mp = mcnemar_edwards(sf, fs)
        bp = exact_binom_p(min(sf, fs), sf + fs) if (sf + fs) > 0 else 1.0

        chi2_str = f"{chi2:.2f}" if (sf + fs) > 0 else "—"
        mp_str = f"p = {mp:.2e}" if mp < 0.001 else (f"p = {mp:.4f}" if (sf + fs) > 0 else "—")
        bp_str = f"p = {bp:.2e}" if bp < 0.001 else (f"p = {bp:.4f}" if (sf + fs) > 0 else "—")

        print(f"{mname:<20} {ss:>5} {sf:>12} {fs:>12} {ff:>6} {net_shift:>+18} {chi2_str:>14} {mp_str:>14} {bp_str:>14}")

        table3_data.append({
            "model_id": mid,
            "display_name": mname,
            "n_paired": len(common),
            "ss": ss,
            "sf_safer": sf,
            "fs_worse": fs,
            "ff": ff,
            "net_vulnerability_shift": net_shift,
            "mcnemar_chi2": chi2,
            "mcnemar_p": mp,
            "exact_binomial_p": bp,
        })

    # Save comprehensive audited summary
    output_payload = {
        "table1_cross_precision_asr": table1_data,
        "table2_reasoning_length_engagement": table2_data,
        "table3_matched_pair_contingency": table3_data,
        "cell_metrics": cell_metrics,
    }
    # Top-level backward compatibility mapping
    for k, v in cell_metrics.items():
        output_payload[k] = v

    out_file = os.path.join(BASE_DIR, "exp3c_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[+] Successfully exported complete audited metrics to: {out_file}")

if __name__ == "__main__":
    analyze_exp3c()
