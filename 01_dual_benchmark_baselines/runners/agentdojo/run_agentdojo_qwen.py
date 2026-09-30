"""
AgentDojo Benchmark Runner for Qwen3.5-4B via vLLM

Architecture:
  1. vLLM serves Qwen3.5-4B as an OpenAI-compatible API server on localhost
  2. AgentDojo connects to it via the "local" model type (prompt-based tool calling)
     or "vllm_parsed" (native OpenAI function calling, if supported by the model)

Usage:
  # Step 1: Start vLLM server (in a separate terminal)
  vllm serve Qwen/Qwen3.5-4B --dtype float16 --gpu-memory-utilization 0.95 --max-model-len 65536 --port 8000 --language-model-only --reasoning-parser qwen3

  # Step 2: Run this script
  python run_agentdojo_qwen.py

  # Or use the CLI directly:
  python -m agentdojo.scripts.benchmark --model local --model-id Qwen/Qwen3.5-4B --attack tool_knowledge
"""

import os
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr so rich emoji shortcodes (:book:, :wrench:, etc.)
# emitted by agentdojo.logging don't crash on Windows cp1252 consoles.
# errors='replace' ensures unmappable chars become '?' instead of raising.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests
from rich.logging import RichHandler
from rich import print

import openai
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import SuiteResults, benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger, TraceLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suites, get_suite
from qwen_llm import QwenLocalLLM

# -- Configuration --------------------------------------------------------------
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "/mnt/d/hf_cache/Qwen3.5-4B")
VLLM_PORT = int(os.getenv("LOCAL_LLM_PORT", 8001))
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"
LOGDIR = Path(os.getenv("AGENTDOJO_LOGDIR", str(Path(__file__).parent / "results" / "agentdojo" / "qwen")))
BENCHMARK_VERSION = "v1.2.2"

# "local" = prompt-based tool calling (works with any model)
# "vllm_parsed" = native OpenAI function calling (requires model support)
MODEL_TYPE = os.getenv("AGENTDOJO_MODEL_TYPE", "local")


def check_vllm_server() -> bool:
    """Check if the vLLM server is running and reachable."""
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
        resp.raise_for_status()
        models = resp.json()
        model_ids = [m["id"] for m in models.get("data", [])]
        print(f"[green]vLLM server is running on port {VLLM_PORT}[/green]")
        print(f"Available models: {model_ids}")
        return True
    except requests.ConnectionError:
        print(f"[red]vLLM server is NOT running on port {VLLM_PORT}[/red]")
        print(f"Start it with: bash /mnt/c/Users/T2530917/lyceum/A6000/toggle-hijacking/scripts/start_qwen.sh")
        return False
    except Exception as e:
        print(f"[red]Error checking vLLM server: {e}[/red]")
        return False


def _make_pipeline(defense: str | None = None, enable_thinking: bool = False, seed: int | None = None, temperature: float = 0.0) -> AgentPipeline:
    """Build an AgentPipeline using QwenLocalLLM via the official from_config() method."""
    client = openai.OpenAI(api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1")
    llm = QwenLocalLLM(client, MODEL_NAME, temperature=temperature, top_p=0.9, enable_thinking=enable_thinking, seed=seed)
    # Get default system message from a standard config
    default_system_msg = PipelineConfig(
        llm="local", model_id=MODEL_NAME, defense=None,
        system_message_name=None, system_message=None,
    ).system_message
    config = PipelineConfig(
        llm=llm,
        model_id=MODEL_NAME,
        defense=defense,
        system_message_name=None,
        system_message=default_system_msg,
    )
    pipeline = AgentPipeline.from_config(config)
    pipeline.name = "local"
    # Increase max_iters to allow more tool calls (needed for injection attacks)
    for elem in pipeline.elements:
        if hasattr(elem, 'max_iters'):
            elem.max_iters = 30
    return pipeline


_PHASE_STATE = {"phase": "utility", "suite": None, "logdir": None, "llm": None, "enable_thinking": False, "temperature": 0.0, "quant": None}
_TRACELOGGER_PATCHED = False


def _build_task_summary(calls: list[dict]) -> dict:
    total_prompt = sum(m.get("prompt_tokens") or 0 for m in calls)
    total_completion = sum(m.get("completion_tokens") or 0 for m in calls)
    total_latency = sum(m.get("latency_seconds") or 0 for m in calls)
    return {
        "calls": calls,
        "summary": {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_latency_seconds": round(total_latency, 3),
            "avg_tokens_per_second": round(total_completion / total_latency, 2) if total_latency > 0 else None,
            "num_turns": len(calls),
            "num_tool_calls": sum(m.get("num_tool_calls") or 0 for m in calls),
        },
    }


def _flush_task_metrics(task_id: str):
    """Atomically append one task's metrics to the on-disk metrics file."""
    llm = _PHASE_STATE["llm"]
    suite = _PHASE_STATE["suite"]
    logdir = _PHASE_STATE["logdir"]
    phase = _PHASE_STATE["phase"]
    if llm is None or suite is None or logdir is None or not task_id:
        return
    calls = llm.task_metrics.get(task_id) or []
    if not calls:
        return

    metrics_file = logdir / f"local_{suite}_{phase}_metrics.json"

    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = None
    else:
        data = None

    if data is None:
        data = {
            "model": MODEL_NAME,
            "model_config": {
                "temperature": _PHASE_STATE["temperature"],
                "top_p": 0.9,
                "dtype": "bfloat16",
                "max_model_len": 65536,
                "enable_thinking": _PHASE_STATE["enable_thinking"],
                "quant": _PHASE_STATE.get("quant"),
            },
            "vllm_config": {
                "gpu_memory_utilization": 0.95,
                "port": VLLM_PORT,
            },
            "tasks": {},
        }
    if "tasks" not in data:
        data["tasks"] = {}

    data["tasks"][task_id] = _build_task_summary(calls)

    tmp = metrics_file.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(metrics_file)


def _install_per_task_metrics_hook():
    """Patch TraceLogger so each task's metrics are flushed to disk the moment it finishes.

    On enter: tag the LLM with this task's ID and start a fresh metrics bucket.
    On exit: write that bucket atomically before AgentDojo moves to the next task.
    Skipped tasks (already complete) never enter TraceLogger, so they aren't touched.
    """
    global _TRACELOGGER_PATCHED
    if _TRACELOGGER_PATCHED:
        return

    orig_enter = TraceLogger.__enter__
    orig_exit = TraceLogger.__exit__

    def _task_key_from_context(ctx: dict) -> str:
        ut = ctx.get("user_task_id") or ""
        it = ctx.get("injection_task_id")
        return f"{ut}__{it}" if it else ut

    def patched_enter(self):
        result = orig_enter(self)
        llm = _PHASE_STATE["llm"]
        if llm is not None:
            key = _task_key_from_context(self.context)
            llm._current_task_id = key
            llm.task_metrics[key] = []
        return result

    def patched_exit(self, exc_type, exc_value, traceback):
        try:
            return orig_exit(self, exc_type, exc_value, traceback)
        finally:
            llm = _PHASE_STATE["llm"]
            if llm is not None:
                key = _task_key_from_context(self.context)
                try:
                    _flush_task_metrics(key)
                except Exception as e:
                    print(f"  [yellow]Per-task metrics flush failed for {key}: {e}[/yellow]")

    TraceLogger.__enter__ = patched_enter
    TraceLogger.__exit__ = patched_exit
    _TRACELOGGER_PATCHED = True


def benchmark_utility_only(
    suites: list[str] | None = None,
    user_tasks: tuple[str, ...] = (),
    force_rerun: bool = False,
    enable_thinking: bool = False,
    logdir: Path = LOGDIR,
    seed: int | None = None,
    temperature: float = 0.0,
) -> dict[str, SuiteResults]:
    """Run benchmark WITHOUT attacks -- measures utility only (can the agent solve tasks?)."""
    os.environ["LOCAL_LLM_PORT"] = str(VLLM_PORT)

    if suites is None:
        suites = list(get_suites(BENCHMARK_VERSION).keys())

    pipeline = _make_pipeline(enable_thinking=enable_thinking, seed=seed, temperature=temperature)
    llm_elem = next(e for e in pipeline.elements if isinstance(e, QwenLocalLLM))

    _install_per_task_metrics_hook()
    _PHASE_STATE["llm"] = llm_elem
    _PHASE_STATE["logdir"] = logdir
    _PHASE_STATE["enable_thinking"] = enable_thinking
    _PHASE_STATE["temperature"] = temperature
    _PHASE_STATE["phase"] = "utility"

    all_results = {}
    for suite_name in suites:
        suite = get_suite(BENCHMARK_VERSION, suite_name)
        print(f"\n[bold blue]-- Utility Benchmark: {suite_name} --[/bold blue]")

        llm_elem.task_metrics = {}
        _PHASE_STATE["suite"] = suite_name

        with OutputLogger(str(logdir)):
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                user_tasks=user_tasks if user_tasks else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=BENCHMARK_VERSION,
            )

        all_results[suite_name] = results
        _print_utility_results(suite_name, results)
        print(f"  [dim]Per-task metrics: {logdir / f'local_{suite_name}_utility_metrics.json'}[/dim]")

    return all_results


def benchmark_with_attacks(
    attack: str = "tool_knowledge",
    suites: list[str] | None = None,
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    defense: str | None = None,
    force_rerun: bool = False,
    enable_thinking: bool = False,
    logdir: Path = LOGDIR,
    seed: int | None = None,
    temperature: float = 0.0,
) -> dict[str, SuiteResults]:
    """Run benchmark WITH attacks -- measures both utility and security."""
    os.environ["LOCAL_LLM_PORT"] = str(VLLM_PORT)

    if suites is None:
        suites = list(get_suites(BENCHMARK_VERSION).keys())

    pipeline = _make_pipeline(defense=defense, enable_thinking=enable_thinking, seed=seed, temperature=temperature)
    llm_elem = next(e for e in pipeline.elements if isinstance(e, QwenLocalLLM))

    _install_per_task_metrics_hook()
    _PHASE_STATE["llm"] = llm_elem
    _PHASE_STATE["logdir"] = logdir
    _PHASE_STATE["enable_thinking"] = enable_thinking
    _PHASE_STATE["temperature"] = temperature
    _PHASE_STATE["phase"] = "attack"

    all_results = {}
    for suite_name in suites:
        suite = get_suite(BENCHMARK_VERSION, suite_name)
        print(f"\n[bold blue]-- Security Benchmark: {suite_name} (attack={attack}) --[/bold blue]")

        llm_elem.task_metrics = {}
        _PHASE_STATE["suite"] = suite_name
        attacker = load_attack(attack, suite, pipeline)

        with OutputLogger(str(logdir)):
            results = benchmark_suite_with_injections(
                pipeline,
                suite,
                attacker,
                user_tasks=user_tasks if user_tasks else None,
                injection_tasks=injection_tasks if injection_tasks else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=BENCHMARK_VERSION,
            )

        all_results[suite_name] = results
        _print_security_results(suite_name, results)
        print(f"  [dim]Per-task metrics: {logdir / f'local_{suite_name}_attack_metrics.json'}[/dim]")

    return all_results


def report_infra_failures(logdir: Path) -> None:
    """Scan all result JSONs under logdir and print an infrastructure-failure summary.

    A result JSON has `error` set whenever AgentDojo's BadRequestError/ApiError/ServerError
    handler fired (most often: context_length_exceeded). These are infra problems, not
    model task failures, and should not be counted against the model's safety/utility.
    """
    import collections

    if not logdir.exists():
        return

    by_category: dict[str, int] = collections.Counter()
    by_suite_attack: dict[tuple, int] = collections.Counter()
    examples: list[str] = []
    total_results = 0
    total_failures = 0

    for f in logdir.rglob("*.json"):
        if f.name.endswith("_metrics.json"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict) or "suite_name" not in d:
            continue
        total_results += 1
        err = d.get("error")
        if not err:
            continue
        total_failures += 1
        msg = str(err).lower()
        if "context_length_exceeded" in msg or "context" in msg and "exceed" in msg:
            cat = "context_overflow"
        elif "internal server error" in msg or "server error" in msg:
            cat = "server_error"
        elif "timeout" in msg:
            cat = "timeout"
        elif "rate limit" in msg or "rate_limit" in msg:
            cat = "rate_limit"
        else:
            cat = "other"
        by_category[cat] += 1
        by_suite_attack[(d.get("suite_name", "?"), d.get("attack_type") or "none")] += 1
        if len(examples) < 3:
            examples.append(f"  {d.get('suite_name')}/{d.get('user_task_id')}/{d.get('attack_type')}/{d.get('injection_task_id') or 'none'}: {str(err)[:120]}")

    if total_results == 0:
        return
    rate = (total_failures / total_results) * 100
    print(f"\n[bold green]== Infrastructure Failure Report ==[/bold green]")
    print(f"  Scanned: {total_results} result JSONs under {logdir}")
    print(f"  Infra failures: {total_failures} ({rate:.2f}%)")
    if total_failures:
        print(f"  By category: {dict(by_category)}")
        print(f"  By (suite, attack):")
        for (suite, attack), n in sorted(by_suite_attack.items()):
            print(f"    {suite} / {attack}: {n}")
        print(f"  Sample errors:")
        for ex in examples:
            print(ex)


def _print_utility_results(suite_name: str, results: SuiteResults):
    utility_results = results["utility_results"].values()
    avg_utility = sum(utility_results) / len(utility_results) if utility_results else 0
    print(f"  [bold]Suite:[/bold] {suite_name}")
    print(f"  [bold]Average Utility:[/bold] {avg_utility * 100:.2f}%")


def _print_security_results(suite_name: str, results: SuiteResults):
    utility_results = results["utility_results"].values()
    avg_utility = sum(utility_results) / len(utility_results) if utility_results else 0

    passed_injection = sum(results["injection_tasks_utility_results"].values())
    total_injection = len(results["injection_tasks_utility_results"])

    security_results = list(results["security_results"].values())
    # In AgentDojo, security=True means the injection task SUCCEEDED (attack won).
    # Report as attack success rate; safety is its complement.
    attack_success_rate = sum(security_results) / len(security_results) if security_results else 0

    print(f"  [bold]Suite:[/bold] {suite_name}")
    print(f"  [bold]Average Utility:[/bold]        {avg_utility * 100:.2f}%")
    print(f"  [bold]Injection Tasks Passed:[/bold]  {passed_injection}/{total_injection}")
    print(f"  [bold]Attack Success Rate:[/bold]     {attack_success_rate * 100:.2f}%")
    print(f"  [bold]Safety Rate:[/bold]             {(1 - attack_success_rate) * 100:.2f}%")


def print_combined_summary(all_results: dict[str, SuiteResults], with_security: bool = False):
    """Print a combined summary across all suites."""
    print("\n[bold green]===================================================[/bold green]")
    print(f"[bold green]  AgentDojo Results for {MODEL_NAME}[/bold green]")
    print(f"[bold green]  Model Type: {MODEL_TYPE}[/bold green]")
    print("[bold green]===================================================[/bold green]")

    combined_utility = {}
    combined_security = {}

    for suite_name, results in all_results.items():
        for key, val in results["utility_results"].items():
            combined_utility[(suite_name + "_" + key[0], key[1])] = val
        if with_security:
            for key, val in results["security_results"].items():
                combined_security[(suite_name + "_" + key[0], key[1])] = val

    avg_utility = sum(combined_utility.values()) / len(combined_utility) if combined_utility else 0
    print(f"\n  [bold]Overall Average Utility:[/bold]  {avg_utility * 100:.2f}%")

    if with_security and combined_security:
        attack_success_rate = sum(combined_security.values()) / len(combined_security)
        print(f"  [bold]Overall Attack Success Rate:[/bold]  {attack_success_rate * 100:.2f}%")
        print(f"  [bold]Overall Safety Rate:[/bold]          {(1 - attack_success_rate) * 100:.2f}%")

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AgentDojo Benchmark for Qwen3.5 via vLLM")
    parser.add_argument("--quick", action="store_true", help="Quick test: 5 samples from workspace suite only")
    parser.add_argument("--suite", "-s", type=str, default=None, help="Suite name (workspace, travel, banking, slack)")
    parser.add_argument("--attack", "--attacks", dest="attacks", nargs="+", default=None,
                        help="One or more attacks to run (e.g. --attacks tool_knowledge important_instructions). Default: tool_knowledge.")
    parser.add_argument("--utility-only", action="store_true", help="Run utility benchmark only (no attacks)")
    parser.add_argument("--security-only", action="store_true", help="Run only the attack/security benchmark; do not rerun or rewrite utility metrics")
    parser.add_argument("--force-rerun", "-f", action="store_true", help="Force rerun already completed tasks")
    parser.add_argument("--think", action="store_true", help="Enable Qwen3.5 thinking/reasoning mode")
    parser.add_argument("--user-tasks", nargs="+", default=None, help="Specific user task IDs to run (e.g. user_task_26)")
    parser.add_argument("--injection-tasks", nargs="+", default=None, help="Specific injection task IDs to run (e.g. injection_task_0)")
    parser.add_argument("--seed", type=int, default=None, help="Number of seeds to run (e.g. --seed 3 -> seeds 0,1,2 in seed_*/ subfolders). Omit for single un-seeded run (legacy paths).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default 0.0 = greedy). Set >0 (e.g. 0.7) for genuine seed-based variance across multi-seed runs.")
    parser.add_argument("--quant", type=str, default=None,
                        help="Quantization label for output isolation (e.g. fp16, fp8, awq_int4, nf4). "
                             "When set, results land in <LOGDIR>/<quant>/ instead of <LOGDIR>/. "
                             "Combines with --seed: <LOGDIR>/<quant>/seed_X/. "
                             "Recorded in metrics file's model_config for traceability. "
                             "Omit for legacy paths.")
    args = parser.parse_args()

    if args.utility_only and args.security_only:
        parser.error("--utility-only and --security-only cannot be used together")
    if args.seed is not None and args.seed < 1:
        parser.error("--seed must be a positive integer (number of seeds to run)")
    if args.temperature < 0.0 or args.temperature > 2.0:
        parser.error("--temperature must be in [0.0, 2.0]")
    if args.quant is not None:
        import re as _re
        if not _re.fullmatch(r"[A-Za-z0-9._-]+", args.quant):
            parser.error("--quant must contain only [A-Za-z0-9._-] (used as a directory name)")

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )

    if not check_vllm_server():
        print(f"\n[yellow]Start vLLM first:[/yellow]")
        print(f"  bash /mnt/c/Users/T2530917/lyceum/A6000/toggle-hijacking/scripts/start_qwen.sh")
        sys.exit(1)

    print(f"\n[bold]Model:[/bold] {MODEL_NAME}  |  [bold]Type:[/bold] {MODEL_TYPE}  |  [bold]Log:[/bold] {LOGDIR}")

    if args.user_tasks:
        user_tasks = tuple(args.user_tasks)
        suites = [args.suite] if args.suite else None
        print(f"[bold cyan]Running specific tasks: {user_tasks}[/bold cyan]")
    elif args.quick:
        suites = ["workspace"]
        user_tasks = tuple(f"user_task_{i}" for i in range(5))
        print("[bold cyan]QUICK TEST: 5 workspace samples[/bold cyan]")
    else:
        suites = [args.suite] if args.suite else None
        user_tasks = ()

    if args.think:
        print("[bold yellow]  Thinking mode: ENABLED[/bold yellow]")
    print(f"[bold yellow]  Temperature: {args.temperature}[/bold yellow]")
    if args.seed is not None and args.temperature == 0.0:
        print("[yellow]  Note: temperature=0.0 makes seeds nearly redundant (greedy decoding). Consider --temperature 0.7 for real variance.[/yellow]")

    # Quant scoping: --quant LABEL pushes everything into <LOGDIR>/<quant>/ so results
    # from different precisions (fp16, fp8, awq_int4, nf4) don't overwrite each other.
    base_logdir = LOGDIR / args.quant if args.quant else LOGDIR
    if args.quant:
        print(f"[bold yellow]  Quant scope: {args.quant} -> {base_logdir}[/bold yellow]")
        _PHASE_STATE["quant"] = args.quant

    # Seed sweep: if --seed N given, run N seeds (0..N-1) into seed_*/ subfolders.
    # If --seed omitted, run once with base_logdir unchanged and no fixed seed.
    if args.seed is None:
        seed_runs = [(None, base_logdir)]
    else:
        seed_runs = [(s, base_logdir / f"seed_{s}") for s in range(args.seed)]
        print(f"[bold yellow]  Multi-seed run: {args.seed} seeds (0..{args.seed - 1}) -> {base_logdir}/seed_*[/bold yellow]")

    if args.attacks:
        attacks = list(args.attacks)
    elif args.utility_only or args.quick:
        attacks = []
    else:
        attacks = ["tool_knowledge"]

    # Validate attack names against the registry up front
    if attacks:
        unknown = [a for a in attacks if a not in ATTACKS]
        if unknown:
            print(f"[red]Unknown attack(s): {unknown}. Available: {sorted(ATTACKS.keys())}[/red]")
            sys.exit(2)

    for seed_value, run_logdir in seed_runs:
        if seed_value is not None:
            run_logdir.mkdir(parents=True, exist_ok=True)
            print(f"\n[bold magenta]====== Seed {seed_value} -> {run_logdir} ======[/bold magenta]")

        if not args.security_only:
            print("\n[bold cyan]Phase 1: Utility Benchmark[/bold cyan]")
            utility_results = benchmark_utility_only(
                suites=suites,
                user_tasks=user_tasks,
                force_rerun=args.force_rerun,
                enable_thinking=args.think,
                logdir=run_logdir,
                seed=seed_value,
                temperature=args.temperature,
            )
            print_combined_summary(utility_results, with_security=False)

        for attack in attacks:
            print(f"\n[bold cyan]Phase 2: Security Benchmark (attack={attack})[/bold cyan]")
            security_results = benchmark_with_attacks(
                attack=attack,
                suites=suites,
                user_tasks=user_tasks,
                injection_tasks=tuple(args.injection_tasks) if args.injection_tasks else (),
                force_rerun=args.force_rerun,
                enable_thinking=args.think,
                logdir=run_logdir,
                seed=seed_value,
                temperature=args.temperature,
            )
            print_combined_summary(security_results, with_security=True)

        report_infra_failures(run_logdir)


if __name__ == "__main__":
    main()
