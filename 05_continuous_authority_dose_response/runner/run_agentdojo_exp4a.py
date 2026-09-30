"""
EXP4A AgentDojo Runner — Dose-Response on AgentDojo.

Tests RQ4: How little prompt change is needed to flip a "0% ASR" model?
L0=baseline (use existing data), L1-L3=increasing dose, L4=full reframe.

Usage:
  cd /home/user1/lyceum/thesis/EXP4A/runner
  python run_agentdojo_exp4a.py --model qwen --condition l1 --security-only
  python run_agentdojo_exp4a.py --model qwen --condition l2 --security-only
  python run_agentdojo_exp4a.py --model qwen --condition l3 --security-only
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "00_Baseline_Sweep", "runners", "agentdojo"))
sys.path.insert(0, "/home/user1/lyceum/thesis/AgentDojoRunner")
sys.path.insert(0, os.path.dirname(__file__))

from agentdojo_exp_base import *
from agentdojo_exp_prompts import get_system_message

def main():
    import argparse
    parser = argparse.ArgumentParser(description="EXP4A AgentDojo: Dose-Response")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--condition", required=True, choices=["l0", "l1", "l2", "l3", "l4"])
    parser.add_argument("--quant", default="fp16")
    parser.add_argument("--suite", "-s", type=str, default=None)
    parser.add_argument("--security-only", action="store_true")
    parser.add_argument("--force-rerun", "-f", action="store_true")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--user-tasks", nargs="+", default=None)
    parser.add_argument("--injection-tasks", nargs="+", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-base", default="/home/user1/lyceum/thesis/EXP4A/results_agentdojo")
    args = parser.parse_args()

    import logging
    from rich.logging import RichHandler
    logging.basicConfig(format="%(message)s", level=logging.INFO, datefmt="%H:%M:%S", handlers=[RichHandler(show_path=False, markup=True)])

    if not check_vllm_server():
        sys.exit(1)

    model_name = MODEL_REGISTRY[args.model]["model_name"]
    think_label = "thinkON" if args.think else "thinkOFF"
    cond_dir = f"{args.condition}_{args.quant}_{think_label}"
    logdir = Path(args.output_base) / args.model / cond_dir
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"\n[bold]EXP4A AgentDojo: {model_name} | dose={args.condition} | think={args.think} | output={logdir}[/bold]")

    suites = [args.suite] if args.suite else None
    user_tasks = tuple(args.user_tasks) if args.user_tasks else ()
    injection_tasks = tuple(args.injection_tasks) if args.injection_tasks else ()

    for attack in ["tool_knowledge"]:
        security_results = benchmark_with_attacks(
            model_key=args.model, condition=args.condition, system_message=get_system_message(args.condition), attack=attack,
            suites=suites, user_tasks=user_tasks, injection_tasks=injection_tasks,
            force_rerun=args.force_rerun, enable_thinking=args.think,
            logdir=logdir, temperature=args.temperature,
        )
    report_infra_failures(logdir)

if __name__ == "__main__":
    main()
