"""
EXP5A AgentDojo Runner — Thinking ON/OFF × 4 Conditions on AgentDojo.

Tests RQ5: Does reasoning defend against prompt injection?
4 conditions: control, reframe, anomaly, exp2a
2 think states: ON, OFF (reasoning models only; Hermes = OFF only)
6 models: nanbeige, lfm, nano, gemma, qwen, hermes

Usage:
  cd /home/user1/lyceum/thesis/EXP5A/runner
  python run_agentdojo_exp5a.py --model gemma --condition reframe --think --security-only
  python run_agentdojo_exp5a.py --model gemma --condition reframe --security-only
  python run_agentdojo_exp5a.py --model hermes --condition control --security-only
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "00_Baseline_Sweep", "runners", "agentdojo"))
sys.path.insert(0, "/home/user1/lyceum/thesis/AgentDojoRunner")
sys.path.insert(0, os.path.dirname(__file__))

from agentdojo_exp_base import *
from agentdojo_exp_prompts import get_system_message

# Models that support thinking toggle (native reasoning token output)
REASONING_MODELS = {"nanbeige", "nano", "gemma", "qwen", "qwen9b", "gemma12b", "ministral", "ornith"}
# Non-reasoning models without native CoT tokens (always evaluated thinkOFF)
NON_REASONING_MODELS = {"minicpm5", "spark", "lfm"}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="EXP5A AgentDojo: Thinking ON/OFF × 4 Conditions")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--condition", required=True, choices=["control", "reframe", "anomaly", "exp2a"])
    parser.add_argument("--quant", default="fp16")
    parser.add_argument("--suite", "-s", type=str, default=None)
    parser.add_argument("--security-only", action="store_true")
    parser.add_argument("--force-rerun", "-f", action="store_true")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--user-tasks", nargs="+", default=None)
    parser.add_argument("--injection-tasks", nargs="+", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-base", default="/home/user1/lyceum/thesis/EXP5A/results_agentdojo")
    args = parser.parse_args()

    import logging
    from rich.logging import RichHandler
    logging.basicConfig(format="%(message)s", level=logging.INFO, datefmt="%H:%M:%S", handlers=[RichHandler(show_path=False, markup=True)])

    if not check_vllm_server():
        sys.exit(1)

    # Hermes is non-reasoning — force thinkOFF
    if args.model == "hermes" and args.think:
        print("[yellow]Hermes is non-reasoning — ignoring --think flag (always thinkOFF)[/yellow]")
        args.think = False

    model_name = MODEL_REGISTRY[args.model]["model_name"]
    think_label = "thinkON" if args.think else "thinkOFF"
    cond_dir = f"{args.condition}_{think_label}"
    logdir = Path(args.output_base) / args.model / cond_dir
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"\n[bold]EXP5A AgentDojo: {model_name} | condition={args.condition} | think={think_label} | output={logdir}[/bold]")

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
