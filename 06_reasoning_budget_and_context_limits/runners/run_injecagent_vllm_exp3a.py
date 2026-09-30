#!/usr/bin/env python3
"""
EXP 3A Runner: Context and Reasoning Budget Invariance.

Evaluates InjecAgent under explicit completion token budget constraints:
  - 16,000 tokens (loose budget)
  - 8,000 tokens (moderate budget)
  - 4,000 tokens (tight budget)
  - 2,000 / 1,000 / 512 / 256 tokens (aggressive truncation for long-thinking models)
  - Baseline (60,000 tokens, unconstrained)

Usage:
  python run_injecagent_vllm_exp3a.py --model qwen --budget 4000 --precision fp16 --output_dir ./raw_results/qwen/tokens_4000
"""

import os
import sys
import argparse
import subprocess

# Standard 11-Model Registry
MODEL_CONFIGS = {
    "nanbeige": {
        "model_name": "/home/user1/hf_cache/Nanbeige4.2-3B",
        "display_name": "Nanbeige4.2",
    },
    "lfm": {
        "model_name": "/home/user1/hf_cache/LFM2.5-2.6B",
        "display_name": "LFM2.5",
    },
    "nano": {
        "model_name": "/home/user1/hf_cache/Nemotron-3-Nano-4B",
        "display_name": "Nemotron3Nano4B",
    },
    "gemma": {
        "model_name": "/home/user1/hf_cache/gemma-4-E4B-it",
        "display_name": "Gemma4E4B",
    },
    "qwen": {
        "model_name": "/home/user1/hf_cache/Qwen3.5-4B",
        "display_name": "Qwen3.5",
    },
    "qwen9b": {
        "model_name": "/mnt/e/Models/Qwen3.5-9B",
        "display_name": "Qwen3.5-9B",
    },
    "ornith": {
        "model_name": "/mnt/e/Models/Ornith-1.5-9B",
        "display_name": "Ornith1.5-9B",
    },
    "minicpm5": {
        "model_name": "/home/user1/hf_cache/MiniCPM5-2B",
        "display_name": "MiniCPM5-2B",
    },
    "spark": {
        "model_name": "spark25",
        "display_name": "Spark-X2.5-4B",
    },
    "gemma12b": {
        "model_name": "/mnt/e/Models/gemma-4-12B-it",
        "display_name": "Gemma4-12B",
    },
    "ministral": {
        "model_name": "/mnt/e/Models/Ministral-3-14B-Reasoning",
        "display_name": "Ministral-3-14B",
    },
}

SUPPORTED_BUDGETS = [16000, 8000, 4000, 2000, 1000, 512, 256]

def parse_args():
    parser = argparse.ArgumentParser(description="EXP 3A: Reasoning Budget Invariance Runner")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()), help="Model identifier")
    parser.add_argument("--budget", required=True, type=int, choices=SUPPORTED_BUDGETS, help="Token budget constraint")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp8", "nf4"], help="Model precision")
    parser.add_argument("--output_dir", required=True, help="Path to save evaluation output JSON files")
    parser.add_argument("--limit", type=int, default=0, help="Test case limit (0 = all cases, default 206 for budget sweep)")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent worker threads")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"=== Running EXP 3A: Model={args.model} | Budget={args.budget} | Precision={args.precision} ===")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Locate base InjecAgent runner
    base_runner = os.path.join(os.path.dirname(__file__), "..", "..", "00_Baseline_Sweep", "runners", "injecagent", "run_injecagent_vllm.py")
    if not os.path.exists(base_runner):
        base_runner = "/mnt/c/Users/T2530917/lyceum/A6000/thesis/injecagent_src/run_injecagent_vllm.py"
    
    cmd = [
        sys.executable, base_runner,
        "--model", args.model,
        "--precision", args.precision,
        "--output_dir", args.output_dir,
        "--max_tokens", str(args.budget),
        "--workers", str(args.workers),
    ]
    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])
        
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
