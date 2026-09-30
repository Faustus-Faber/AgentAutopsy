#!/usr/bin/env python3
"""
EXP 3C Runner: Cross-Precision Evaluation Sweep.

Orchestrates cross-precision evaluations across FP16, FP8, and NF4 quantizations
to measure whether quantization degrades agentic robustness, moderates reasoning lengths,
or causes precision dysfunction.

Usage:
  python run_precision_sweep.py --model qwen --precision fp8 --benchmark injecagent
  python run_precision_sweep.py --model nano --precision nf4 --benchmark agentdojo
"""

import os
import sys
import argparse
import subprocess

MODEL_CONFIGS = {
    "nanbeige": {"display_name": "Nanbeige4.2-3B"},
    "lfm": {"display_name": "LFM2.5-2.6B"},
    "nano": {"display_name": "Nemotron-3-Nano-4B"},
    "gemma": {"display_name": "Gemma-4-E4B-it"},
    "qwen": {"display_name": "Qwen3.5-4B"},
    "qwen9b": {"display_name": "Qwen3.5-9B"},
    "ornith": {"display_name": "Ornith-1.5-9B"},
    "minicpm5": {"display_name": "MiniCPM5-2B"},
    "spark": {"display_name": "Spark-X2.5-4B"},
    "gemma12b": {"display_name": "Gemma4-12B-it"},
    "ministral": {"display_name": "Ministral-3-14B-Reasoning"},
}

PRECISIONS = ["fp16", "fp8", "nf4"]
BENCHMARKS = ["injecagent", "agentdojo"]

def parse_args():
    parser = argparse.ArgumentParser(description="EXP 3C: Cross-Precision Sweep Runner")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()), help="Model to evaluate")
    parser.add_argument("--precision", required=True, choices=PRECISIONS, help="Precision quantization level")
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS, help="Target evaluation benchmark")
    parser.add_argument("--output-base", default="./results", help="Base output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"=== EXP 3C Precision Sweep: {args.model} | {args.precision} | {args.benchmark} ===")
    
    if args.benchmark == "injecagent":
        runner = os.path.join(os.path.dirname(__file__), "..", "..", "00_Baseline_Sweep", "runners", "injecagent", "run_injecagent_vllm.py")
        out_dir = os.path.join(args.output_base, "injecagent", args.model, args.precision)
        cmd = [sys.executable, runner, "--model", args.model, "--precision", args.precision, "--output_dir", out_dir]
    else:
        runner = os.path.join(os.path.dirname(__file__), "..", "..", "00_Baseline_Sweep", "runners", "agentdojo", "run_agentdojo_vllm.py")
        cmd = [sys.executable, runner, "--model", args.model, "--quant", args.precision, "--think"]
        
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
