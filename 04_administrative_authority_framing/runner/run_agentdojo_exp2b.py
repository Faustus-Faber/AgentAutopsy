"""
EXP2B AgentDojo Runner — Reframe-forced vs Anomaly-forced on AgentDojo.

Tests RQ2: Is task reframe the causal mechanism of injection compliance?
Runs on AgentDojo (4 suites, 949 injection cases) to test cross-benchmark generalization.

Usage:
  cd /home/user1/lyceum/thesis/EXP2B/runner
  python run_agentdojo_exp2b.py --model nano --condition reframe --security-only
  python run_agentdojo_exp2b.py --model nano --condition anomaly --security-only
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "00_Baseline_Sweep", "runners", "agentdojo"))
sys.path.insert(0, "/home/user1/lyceum/thesis/AgentDojoRunner")
sys.path.insert(0, os.path.dirname(__file__))

# Import the unified runner and restrict conditions
from agentdojo_exp_base import *
from agentdojo_exp_prompts import get_system_message

# Override main to restrict to EXP2B conditions only
def main():
    import argparse
    parser = argparse.ArgumentParser(description="EXP2B AgentDojo: Reframe vs Anomaly")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--condition", required=True, choices=["control", "reframe", "anomaly"])
    parser.add_argument("--quant", default="fp16")
    parser.add_argument("--suite", "-s", type=str, default=None)
    parser.add_argument("--security-only", action="store_true")
    parser.add_argument("--force-rerun", "-f", action="store_true")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--user-tasks", nargs="+", default=None)
    parser.add_argument("--injection-tasks", nargs="+", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-base", default="/home/user1/lyceum/thesis/EXP2B/results_agentdojo")
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
    print(f"\n[bold]EXP2B AgentDojo: {model_name} | condition={args.condition} | think={args.think} | output={logdir}[/bold]")

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
