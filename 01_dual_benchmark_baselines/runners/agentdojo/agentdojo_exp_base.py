"""
Shared AgentDojo experiment runner module.

This provides the core pipeline-building and benchmark-running logic
used by all experiment runners (EXP2B, EXP4A, EXP5A, EXP6, EXP7).

Each experiment folder has its own runner that imports from this module
and restricts the available conditions.

DO NOT run this file directly — use the per-experiment runners instead:
  EXP2B/runner/run_agentdojo_exp2b.py
  EXP4A/runner/run_agentdojo_exp4a.py
  EXP5A/runner/run_agentdojo_exp5a.py
  EXP6/runner/run_agentdojo_exp6.py
  EXP7_context_ablation/runner/run_context_ablation.py
"""

import os
import json
import logging
import sys
import time
from pathlib import Path

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
from agentdojo.task_suite.load_suites import get_suites, get_suite

# -- Model registry --------------------------------------------------------------
MODEL_REGISTRY = {
    "nanbeige": {"model_name": "/mnt/d/hf_cache/Nanbeige4.2-3B", "llm_class": "native"},
    "lfm": {"model_name": "/mnt/d/hf_cache/LFM2.5-2.6B", "llm_class": "native"},
    "nano": {"model_name": "/mnt/d/hf_cache/Nemotron-3-Nano-4B", "llm_class": "native"},
    "gemma": {"model_name": "/mnt/d/hf_cache/gemma-4-E4B-it", "llm_class": "gemma"},
    "qwen": {"model_name": "/mnt/d/hf_cache/Qwen3.5-4B", "llm_class": "gemma"},
    "qwen9b": {"model_name": "/mnt/e/Models/Qwen3.5-9B", "llm_class": "native"},
    "ornith": {"model_name": "/mnt/e/Models/Ornith-1.5-9B", "llm_class": "native"},
    "minicpm5": {"model_name": "/home/user1/hf_cache/MiniCPM5-2B", "llm_class": "native"},
    "spark": {"model_name": "spark25", "llm_class": "native"},
    "gemma12b": {"model_name": "/mnt/e/Models/gemma-4-12B-it", "llm_class": "gemma"},
    "ministral": {"model_name": "/mnt/e/Models/Ministral-3-14B-Reasoning", "llm_class": "native"},
}

VLLM_PORT = int(os.getenv("LOCAL_LLM_PORT", 8001))
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"
BENCHMARK_VERSION = "v1.2.2"

_PHASE_STATE = {"phase": "utility", "suite": None, "logdir": None, "llm": None,
                "enable_thinking": False, "temperature": 0.0, "quant": None,
                "condition": None, "model_key": None}
_TRACELOGGER_PATCHED = False


def check_vllm_server() -> bool:
    try:
        resp = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
        resp.raise_for_status()
        model_ids = [m["id"] for m in resp.json().get("data", [])]
        print(f"[green]vLLM server running on port {VLLM_PORT}[/green]")
        print(f"Available models: {model_ids}")
        return True
    except Exception as e:
        print(f"[red]vLLM server NOT reachable: {e}[/red]")
        return False


def _get_llm_class(model_key: str):
    cfg = MODEL_REGISTRY[model_key]
    if cfg["llm_class"] == "native":
        from native_llm import NativeFunctionCallingLLM
        return NativeFunctionCallingLLM
    elif cfg["llm_class"] == "hermes":
        from hermes_llm import HermesLocalLLM
        return HermesLocalLLM
    elif cfg["llm_class"] == "gemma":
        from gemma_llm import GemmaLocalLLM
        return GemmaLocalLLM
    raise ValueError(f"Unknown LLM class: {cfg['llm_class']}")


def _make_pipeline(model_key: str, condition: str, system_message: str,
                   enable_thinking: bool = False, seed: int | None = None,
                   temperature: float = 0.0) -> AgentPipeline:
    """Build an AgentPipeline with a custom system message for the experiment condition."""
    cfg = MODEL_REGISTRY[model_key]
    model_name = cfg["model_name"]
    llm_class = cfg["llm_class"]
    client = openai.OpenAI(api_key="EMPTY", base_url=f"http://localhost:{VLLM_PORT}/v1", timeout=1200.0)

    if llm_class == "native":
        from native_llm import NativeFunctionCallingLLM
        llm = NativeFunctionCallingLLM(client, model_name, temperature=temperature, top_p=0.9,
                                        enable_thinking=enable_thinking, seed=seed)
    elif llm_class == "hermes":
        from hermes_llm import HermesLocalLLM
        llm = HermesLocalLLM(client, model_name, temperature=temperature, top_p=0.9,
                              enable_thinking=enable_thinking, seed=seed)
    elif llm_class == "gemma":
        from gemma_llm import GemmaLocalLLM
        llm = GemmaLocalLLM(client, model_name, temperature=temperature, top_p=0.9,
                             enable_thinking=enable_thinking, seed=seed)
    else:
        raise ValueError(f"Unknown LLM class: {llm_class}")

    config = PipelineConfig(
        llm=llm, model_id=model_name, defense=None,
        system_message_name=None, system_message=system_message,
    )
    pipeline = AgentPipeline.from_config(config)
    pipeline.name = "local"
    for elem in pipeline.elements:
        if hasattr(elem, 'max_iters'):
            elem.max_iters = 30
    return pipeline


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
        data = {"model_config": {"model": MODEL_REGISTRY.get(_PHASE_STATE.get("model_key", ""), {}).get("model_name", "?"),
                                  "quant": _PHASE_STATE.get("quant"),
                                  "condition": _PHASE_STATE.get("condition"),
                                  "enable_thinking": _PHASE_STATE.get("enable_thinking")}, "tasks": {}}
    data.setdefault("tasks", {})[task_id] = _build_task_summary(calls)
    tmp = metrics_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(metrics_file)


def _install_per_task_metrics_hook():
    global _TRACELOGGER_PATCHED
    if _TRACELOGGER_PATCHED:
        return
    _TRACELOGGER_PATCHED = True
    try:
        from agentdojo.logging import TraceLogger
        _orig = TraceLogger.log_task_result
        def _patched(self, task_id, *a, **kw):
            try:
                _flush_task_metrics(task_id)
            except Exception:
                pass
            return _orig(self, task_id, *a, **kw)
        TraceLogger.log_task_result = _patched
    except Exception:
        pass


def benchmark_with_attacks(
    model_key: str,
    condition: str,
    system_message: str,
    attack: str = "tool_knowledge",
    suites: list[str] | None = None,
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    force_rerun: bool = False,
    enable_thinking: bool = False,
    logdir: Path | None = None,
    seed: int | None = None,
    temperature: float = 0.0,
) -> dict[str, SuiteResults]:
    """Run security benchmark with a custom system message for the experiment condition."""
    os.environ["LOCAL_LLM_PORT"] = str(VLLM_PORT)
    if suites is None:
        suites = list(get_suites(BENCHMARK_VERSION).keys())

    pipeline = _make_pipeline(model_key, condition, system_message,
                               enable_thinking=enable_thinking, seed=seed, temperature=temperature)
    llm_cls = _get_llm_class(model_key)
    llm_elem = next(e for e in pipeline.elements if isinstance(e, llm_cls))

    _install_per_task_metrics_hook()
    _PHASE_STATE["llm"] = llm_elem
    _PHASE_STATE["logdir"] = logdir
    _PHASE_STATE["enable_thinking"] = enable_thinking
    _PHASE_STATE["temperature"] = temperature
    _PHASE_STATE["phase"] = "attack"
    _PHASE_STATE["condition"] = condition
    _PHASE_STATE["model_key"] = model_key

    all_results = {}
    for suite_name in suites:
        suite = get_suite(BENCHMARK_VERSION, suite_name)
        print(f"\n[bold blue]-- Security: {suite_name} (attack={attack}, condition={condition}) --[/bold blue]")
        llm_elem.task_metrics = {}
        _PHASE_STATE["suite"] = suite_name
        attacker = load_attack(attack, suite, pipeline)
        with OutputLogger(str(logdir)):
            results = benchmark_suite_with_injections(
                pipeline, suite, attacker,
                user_tasks=user_tasks if user_tasks else None,
                injection_tasks=injection_tasks if injection_tasks else None,
                logdir=logdir, force_rerun=force_rerun,
                benchmark_version=BENCHMARK_VERSION,
            )
        all_results[suite_name] = results
        _print_security_results(suite_name, results)
        print(f"  [dim]Metrics: {logdir / f'local_{suite_name}_attack_metrics.json'}[/dim]")
    return all_results


def _print_security_results(suite_name: str, results: SuiteResults):
    security_results = list(results["security_results"].values())
    asr = sum(security_results) / len(security_results) if security_results else 0
    passed = sum(results["injection_tasks_utility_results"].values())
    total = len(results["injection_tasks_utility_results"])
    print(f"  [bold]Suite:[/bold] {suite_name}")
    print(f"  [bold]ASR:[/bold] {asr*100:.2f}%  |  [bold]Injection Passed:[/bold] {passed}/{total}")
    print(f"  [bold]Safety:[/bold] {(1-asr)*100:.2f}%")


def report_infra_failures(logdir: Path) -> None:
    import collections
    if not logdir.exists():
        return
    by_cat: dict[str, int] = collections.Counter()
    total = 0
    failures = 0
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
        total += 1
        err = d.get("error")
        if not err:
            continue
        failures += 1
        msg = str(err).lower()
        if "context_length" in msg or "context" in msg and "exceed" in msg:
            by_cat["context_overflow"] += 1
        elif "server error" in msg:
            by_cat["server_error"] += 1
        elif "timeout" in msg:
            by_cat["timeout"] += 1
        else:
            by_cat["other"] += 1
    if total > 0:
        print(f"\n[bold green]== Infra Failures: {failures}/{total} ({failures/total*100:.2f}%) ==[/bold green]")
        if failures:
            print(f"  By category: {dict(by_cat)}")
