"""EXP3C: Cross-Precision Reasoning Length Analysis.

Does quantization moderate the engagement-compliance relationship?
- Compare reasoning length across FP16/FP8/NF4
- Compare ASR across precisions
- Test if NF4 increases overthinking (longer reasoning → more reframe → higher ASR)
- Per-category breakdown (DH vs DS)
- Matched-pair analysis across precisions
"""
import json, glob, os, sys, statistics
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "00_Baseline_Sweep", "raw_results", "injecagent"))
BASELINE_BASE = DEFAULT_BASE if os.path.isdir(DEFAULT_BASE) else "/home/user1/lyceum/thesis/results/injecagent"

MODELS = {
    "qwen": "Qwen3.5-4B",
    "gemma": "Gemma-4-E4B-it",
    "nano": "Nemotron-3-Nano-4B",
    "nanbeige": "Nanbeige-4.2-3B",
    "lfm": "LFM-2.5-2.6B",
}
PRECISIONS = ["fp16", "fp8", "nf4"]

def load_results(model, precision):
    d = os.path.join(BASELINE_BASE, model, precision)
    if not os.path.isdir(d):
        return {}
    results = {}
    files = [f for f in sorted(glob.glob(os.path.join(d, "*.json"))) if 'summary' not in f and 'metrics' not in f and 'progress' not in f]
    for f in files:
        try:
            data = json.load(open(f, encoding="utf-8", errors="ignore"))
            task_id = data.get("user_task_id", os.path.basename(f).replace(".json",""))
            ev = data.get("eval", "")
            is_succ = (ev == "succ")
            rl = len(data.get("thinking", "") or "") + (len(data.get("thinking_step2", "") or "") if isinstance(data.get("thinking_step2", "") or "", str) else 0)
            cat = data.get("attack_category", "")
            results[task_id] = {
                "succ": is_succ, "eval": ev, "reasoning_len": rl,
                "category": cat,
            }
        except:
            pass
    return results

# ─── 1. ASR and Reasoning Length by Model × Precision ───
print("=" * 130)
print("EXP3C: CROSS-PRECISION REASONING LENGTH ANALYSIS")
print("=" * 130)
print()
print(f"{'Model':<25} | {'Precision':>9} | {'N':>5} | {'ASR':>7} | {'Invalid':>8} | {'Avg RL':>8} | {'Med RL':>8} | {'Max RL':>8} | {'Succ RL':>9} | {'Fail RL':>9} | {'S/F ratio':>10}")
print("-" * 130)

all_data = {}
for mk, name in MODELS.items():
    all_data[mk] = {}
    for prec in PRECISIONS:
        results = load_results(mk, prec)
        if not results:
            print(f"{name:<25} | {prec:>9} | {'N/A':>5} |")
            continue
        
        n = len(results)
        succ = [r for r in results.values() if r["succ"]]
        fail = [r for r in results.values() if not r["succ"]]
        invalid = sum(1 for r in results.values() if r["eval"] == "invalid")
        asr = len(succ) / n * 100 if n > 0 else 0
        
        rls = [r["reasoning_len"] for r in results.values()]
        succ_rls = [r["reasoning_len"] for r in succ]
        fail_rls = [r["reasoning_len"] for r in fail if r["eval"] != "invalid"]
        
        avg_rl = statistics.mean(rls) if rls else 0
        med_rl = statistics.median(rls) if rls else 0
        max_rl = max(rls) if rls else 0
        succ_rl = statistics.mean(succ_rls) if succ_rls else 0
        fail_rl = statistics.mean(fail_rls) if fail_rls else 0
        sf_ratio = succ_rl / fail_rl if fail_rl > 0 else 0
        
        all_data[mk][prec] = {
            "n": n, "asr": asr, "invalid": invalid,
            "avg_rl": avg_rl, "med_rl": med_rl, "max_rl": max_rl,
            "succ_rl": succ_rl, "fail_rl": fail_rl, "sf_ratio": sf_ratio,
            "results": results,
        }
        
        print(f"{name:<25} | {prec:>9} | {n:>5} | {asr:>6.1f}% | {invalid:>8} | {avg_rl:>7.0f} | {med_rl:>7.0f} | {max_rl:>7} | {succ_rl:>8.0f} | {fail_rl:>8.0f} | {sf_ratio:>9.2f}x")
    print()

# ─── 2. ASR Trend Across Precisions ───
print("=" * 130)
print("ASR TREND ACROSS PRECISIONS")
print("=" * 130)
print()
print(f"{'Model':<25} | {'FP16 ASR':>9} | {'FP8 ASR':>9} | {'NF4 ASR':>9} | {'FP16→FP8 Δ':>11} | {'FP16→NF4 Δ':>11} | {'Trend':>12}")
print("-" * 130)

for mk, name in MODELS.items():
    asrs = []
    for prec in PRECISIONS:
        if prec in all_data[mk]:
            asrs.append(all_data[mk][prec]["asr"])
        else:
            asrs.append(None)
    
    fp16, fp8, nf4 = asrs
    if fp16 is not None and nf4 is not None:
        d_fp8 = f"{fp8 - fp16:+.1f}%" if fp8 is not None else "N/A"
        d_nf4 = f"{nf4 - fp16:+.1f}%"
        if nf4 > fp16 + 1:
            trend = "↑ NF4 worse"
        elif nf4 < fp16 - 1:
            trend = "↓ NF4 better"
        else:
            trend = "→ stable"
    else:
        d_fp8 = "N/A"
        d_nf4 = "N/A"
        trend = "N/A"
    
    fp16_s = f"{fp16:.1f}%" if fp16 is not None else "N/A"
    fp8_s = f"{fp8:.1f}%" if fp8 is not None else "N/A"
    nf4_s = f"{nf4:.1f}%" if nf4 is not None else "N/A"
    
    print(f"{name:<25} | {fp16_s:>9} | {fp8_s:>9} | {nf4_s:>9} | {d_fp8:>11} | {d_nf4:>11} | {trend:>12}")

# ─── 3. Reasoning Length Trend Across Precisions ───
print()
print("=" * 130)
print("REASONING LENGTH TREND ACROSS PRECISIONS")
print("=" * 130)
print()
print(f"{'Model':<25} | {'FP16 RL':>9} | {'FP8 RL':>9} | {'NF4 RL':>9} | {'FP16→FP8 Δ':>11} | {'FP16→NF4 Δ':>11} | {'Trend':>12}")
print("-" * 130)

for mk, name in MODELS.items():
    rls = []
    for prec in PRECISIONS:
        if prec in all_data[mk]:
            rls.append(all_data[mk][prec]["avg_rl"])
        else:
            rls.append(None)
    
    fp16, fp8, nf4 = rls
    if fp16 is not None and nf4 is not None:
        d_fp8 = f"{fp8 - fp16:+.0f}" if fp8 is not None else "N/A"
        d_nf4 = f"{nf4 - fp16:+.0f}"
        if nf4 > fp16 * 1.1:
            trend = "↑ NF4 longer"
        elif nf4 < fp16 * 0.9:
            trend = "↓ NF4 shorter"
        else:
            trend = "→ stable"
    else:
        d_fp8 = "N/A"
        d_nf4 = "N/A"
        trend = "N/A"
    
    fp16_s = f"{fp16:.0f}" if fp16 is not None else "N/A"
    fp8_s = f"{fp8:.0f}" if fp8 is not None else "N/A"
    nf4_s = f"{nf4:.0f}" if nf4 is not None else "N/A"
    
    print(f"{name:<25} | {fp16_s:>9} | {fp8_s:>9} | {nf4_s:>9} | {d_fp8:>11} | {d_nf4:>11} | {trend:>12}")

# ─── 4. S/F Ratio Trend (engagement-compliance correlation) ───
print()
print("=" * 130)
print("SUCC/FAIL REASONING RATIO ACROSS PRECISIONS (engagement-compliance correlation)")
print("=" * 130)
print()
print(f"{'Model':<25} | {'FP16 S/F':>9} | {'FP8 S/F':>9} | {'NF4 S/F':>9} | {'Trend':>12} | {'Interpretation':>30}")
print("-" * 130)

for mk, name in MODELS.items():
    ratios = []
    for prec in PRECISIONS:
        if prec in all_data[mk]:
            ratios.append(all_data[mk][prec]["sf_ratio"])
        else:
            ratios.append(None)
    
    fp16, fp8, nf4 = ratios
    if fp16 is not None and nf4 is not None and fp16 > 0 and nf4 > 0:
        if nf4 > fp16 * 1.15:
            trend = "↑ NF4 higher"
            interp = "NF4 amplifies engagement"
        elif nf4 < fp16 * 0.85:
            trend = "↓ NF4 lower"
            interp = "NF4 reduces engagement"
        else:
            trend = "→ stable"
            interp = "Quantization doesn't affect"
    else:
        trend = "N/A"
        interp = "Insufficient data"
    
    fp16_s = f"{fp16:.2f}x" if fp16 is not None and fp16 > 0 else "N/A"
    fp8_s = f"{fp8:.2f}x" if fp8 is not None and fp8 > 0 else "N/A"
    nf4_s = f"{nf4:.2f}x" if nf4 is not None and nf4 > 0 else "N/A"
    
    print(f"{name:<25} | {fp16_s:>9} | {fp8_s:>9} | {nf4_s:>9} | {trend:>12} | {interp:>30}")

# ─── 5. Per-Category (DH vs DS) ASR by Precision ───
print()
print("=" * 130)
print("PER-CATEGORY ASR BY PRECISION")
print("=" * 130)
print()
print(f"{'Model':<25} | {'Category':>10} | {'FP16 ASR':>9} | {'FP8 ASR':>9} | {'NF4 ASR':>9} | {'NF4 vs FP16':>12}")
print("-" * 130)

for mk, name in MODELS.items():
    for cat in ["direct_harm", "data_stealing"]:
        asrs = []
        for prec in PRECISIONS:
            if prec in all_data[mk]:
                cat_results = [r for r in all_data[mk][prec]["results"].values() if r["category"] == cat]
                if cat_results:
                    asrs.append(sum(1 for r in cat_results if r["succ"]) / len(cat_results) * 100)
                else:
                    asrs.append(None)
            else:
                asrs.append(None)
        
        fp16, fp8, nf4 = asrs
        if fp16 is not None and nf4 is not None:
            delta = f"{nf4 - fp16:+.1f}%"
        else:
            delta = "N/A"
        
        fp16_s = f"{fp16:.1f}%" if fp16 is not None else "N/A"
        fp8_s = f"{fp8:.1f}%" if fp8 is not None else "N/A"
        nf4_s = f"{nf4:.1f}%" if nf4 is not None else "N/A"
        cat_label = "DH" if cat == "direct_harm" else "DS"
        
        print(f"{name:<25} | {cat_label:>10} | {fp16_s:>9} | {fp8_s:>9} | {nf4_s:>9} | {delta:>12}")
    print()

# ─── 6. Matched-Pair Analysis: FP16 vs NF4 ───
print("=" * 130)
print("MATCHED-PAIR ANALYSIS: FP16 vs NF4 (case-level outcome changes)")
print("=" * 130)
print()
print(f"{'Model':<25} | {'S→S':>5} | {'S→F':>5} | {'F→S':>5} | {'F→F':>5} | {'Net flipped':>12} | {'ASR change':>11}")
print("-" * 130)

for mk, name in MODELS.items():
    if "fp16" not in all_data[mk] or "nf4" not in all_data[mk]:
        print(f"{name:<25} | N/A")
        continue
    
    fp16_r = all_data[mk]["fp16"]["results"]
    nf4_r = all_data[mk]["nf4"]["results"]
    common = set(fp16_r.keys()) & set(nf4_r.keys())
    
    ss = sum(1 for t in common if fp16_r[t]["succ"] and nf4_r[t]["succ"])
    sf = sum(1 for t in common if fp16_r[t]["succ"] and not nf4_r[t]["succ"])
    fs = sum(1 for t in common if not fp16_r[t]["succ"] and nf4_r[t]["succ"])
    ff = sum(1 for t in common if not fp16_r[t]["succ"] and not nf4_r[t]["succ"])
    
    asr_change = all_data[mk]["nf4"]["asr"] - all_data[mk]["fp16"]["asr"]
    
    print(f"{name:<25} | {ss:>5} | {sf:>5} | {fs:>5} | {ff:>5} | {fs-sf:>+11} | {asr_change:>+10.1f}%")

# ─── 7. Summary: Does quantization moderate engagement-compliance? ───
print()
print("=" * 130)
print("SUMMARY: DOES QUANTIZATION MODERATE THE ENGAGEMENT-COMPLIANCE RELATIONSHIP?")
print("=" * 130)
print()

# Calculate averages
valid_models = [mk for mk in MODELS if "fp16" in all_data[mk] and "nf4" in all_data[mk]]
avg_fp16_asr = statistics.mean([all_data[mk]["fp16"]["asr"] for mk in valid_models])
avg_nf4_asr = statistics.mean([all_data[mk]["nf4"]["asr"] for mk in valid_models])
avg_fp16_rl = statistics.mean([all_data[mk]["fp16"]["avg_rl"] for mk in valid_models])
avg_nf4_rl = statistics.mean([all_data[mk]["nf4"]["avg_rl"] for mk in valid_models])
avg_fp16_sf = statistics.mean([all_data[mk]["fp16"]["sf_ratio"] for mk in valid_models if all_data[mk]["fp16"]["sf_ratio"] > 0])
avg_nf4_sf = statistics.mean([all_data[mk]["nf4"]["sf_ratio"] for mk in valid_models if all_data[mk]["nf4"]["sf_ratio"] > 0])

print(f"  Average ASR:  FP16={avg_fp16_asr:.1f}%  →  NF4={avg_nf4_asr:.1f}%  (Δ={avg_nf4_asr-avg_fp16_asr:+.1f}%)")
print(f"  Average RL:   FP16={avg_fp16_rl:.0f}    →  NF4={avg_nf4_rl:.0f}    (Δ={avg_nf4_rl-avg_fp16_rl:+.0f})")
print(f"  Average S/F:  FP16={avg_fp16_sf:.2f}x   →  NF4={avg_nf4_sf:.2f}x   (Δ={avg_nf4_sf-avg_fp16_sf:+.2f}x)")
print()
print(f"  Models where NF4 increases ASR: {sum(1 for mk in valid_models if all_data[mk]['nf4']['asr'] > all_data[mk]['fp16']['asr'] + 1)}/{len(valid_models)}")
print(f"  Models where NF4 increases RL:  {sum(1 for mk in valid_models if all_data[mk]['nf4']['avg_rl'] > all_data[mk]['fp16']['avg_rl'] * 1.1)}/{len(valid_models)}")
print(f"  Models where NF4 increases S/F: {sum(1 for mk in valid_models if all_data[mk]['nf4']['sf_ratio'] > all_data[mk]['fp16']['sf_ratio'] * 1.15 and all_data[mk]['fp16']['sf_ratio'] > 0)}/{len(valid_models)}")

# Save summary
summary = {}
for mk, name in MODELS.items():
    summary[mk] = {"name": name}
    for prec in PRECISIONS:
        if prec in all_data[mk]:
            d = all_data[mk][prec]
            sf_val = round(d["sf_ratio"], 2) if d["sf_ratio"] > 0 else 0.0
            summary[mk][prec] = {
                "asr": round(d["asr"], 1),
                "avg_rl": round(d["avg_rl"], 0),
                "sf_ratio": sf_val,
                "invalid": d["invalid"],
            }

summary_path = os.path.join(os.path.dirname(__file__), "results_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to: {summary_path}")
