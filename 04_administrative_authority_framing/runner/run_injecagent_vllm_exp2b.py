#!/usr/bin/env python3
"""InjecAgent benchmark runner for local vLLM models.

Adapted from run_injecagent_kimi.py. Uses vLLM's OpenAI-compatible API
to run InjecAgent direct_harm and data_stealing test cases.

Saves full trajectories with reasoning traces (message.reasoning_content)
and tool call trajectories per step.

Usage:
    cd /home/user1/lyceum/thesis/injecagent_src
    python run_injecagent_vllm.py --model nanbeige --precision fp16 \
        --output_dir /home/user1/lyceum/thesis/results/InjecAgent/Nanbeige_FP16
"""
import json
import os
import sys
import time
import argparse
import requests
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

_write_lock = threading.Lock()
_log_lock = threading.Lock()

# ─── Model configurations ───
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

VLLM_BASE = "http://localhost:8001/v1"

# ─── Parse args ───
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
parser.add_argument("--precision", required=True, choices=["fp16", "fp8", "nf4"])
parser.add_argument("--output_dir", required=True, help="Output directory")
parser.add_argument("--mode", default="full", choices=["smoke", "full"])
parser.add_argument("--limit", type=int, default=0, help="Limit total test cases (0 = no limit)")
parser.add_argument("--start", type=int, default=0, help="Start index for case slice (inclusive)")
parser.add_argument("--end", type=int, default=0, help="End index for case slice (exclusive, 0 = to end)")
parser.add_argument("--workers", type=int, default=16, help="Concurrent workers")
parser.add_argument("--max_tokens", type=int, default=60000, help="Max completion tokens")
parser.add_argument("--timeout", type=float, default=1200.0, help="Request timeout seconds")
parser.add_argument("--condition", required=True, choices=["reframe", "anomaly"], help="EXP2B condition: reframe-forced or anomaly-forced")
args = parser.parse_args()

MODEL_NAME = MODEL_CONFIGS[args.model]["model_name"]
DISPLAY_NAME = MODEL_CONFIGS[args.model]["display_name"]
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "run.log")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def save_progress(status, completed, total, phase, extra=None):
    prog = {
        "model": DISPLAY_NAME,
        "precision": args.precision,
        "mode": args.mode,
        "status": status,
        "phase": phase,
        "completed": completed,
        "total": total,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        prog.update(extra)
    with _write_lock:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(prog, f, indent=2)


# ─── Step 1: Check vLLM server ───
log(f"Starting InjecAgent benchmark: {DISPLAY_NAME} {args.precision} ({args.mode})")
log(f"Output dir: {OUTPUT_DIR}")

try:
    resp = requests.get(f"{VLLM_BASE}/models", timeout=10)
    models = resp.json()
    model_ids = [m["id"] for m in models.get("data", [])]
    log(f"OK vLLM reachable, models: {model_ids}")
    if MODEL_NAME not in model_ids:
        log(f"WARNING: Expected model {MODEL_NAME} not in {model_ids}, using first available")
        MODEL_NAME = model_ids[0]
except Exception as e:
    log(f"FAIL Cannot reach vLLM: {e}")
    save_progress("failed", 0, 0, "server_check", {"error": str(e)})
    sys.exit(1)

# ─── Step 2: Import InjecAgent modules ───
# The src/ package lives in /home/user1/lyceum/thesis/injecagent/
# The data/ and runner scripts live in /home/user1/lyceum/thesis/injecagent_src/
# EXP2B: Load condition-specific prompt (reframe-forced or anomaly-forced)
INJECAGENT_PKG = "/home/user1/lyceum/thesis/injecagent"
INJECAGENT_SRC = "/home/user1/lyceum/thesis/injecagent_src"
EXP2B_PROMPTS = "/home/user1/lyceum/thesis/EXP2B/prompts"
sys.path.insert(0, INJECAGENT_PKG)
sys.path.insert(0, EXP2B_PROMPTS)
os.chdir(INJECAGENT_SRC)

from src.utils import get_tool_dict, get_simulated_attacker_tool_response, get_score
from src.output_parsing import evaluate_output_prompted
# EXP2B: Load condition-specific prompt
if args.condition == "reframe":
    from agent_prompts_reframe import PROMPT_DICT
else:
    from agent_prompts_anomaly import PROMPT_DICT

log(f"OK InjecAgent modules imported (EXP2B condition: {args.condition})")

# ─── Step 3: Load test data ───
with open("data/test_cases_dh_base.json") as f:
    dh_data = json.load(f)
with open("data/test_cases_ds_base.json") as f:
    ds_data = json.load(f)

if args.mode == "smoke":
    NUM_DH = 5
    NUM_DS = 3
else:
    NUM_DH = len(dh_data)
    NUM_DS = len(ds_data)

test_cases = []
for i in range(min(NUM_DH, len(dh_data))):
    item = dh_data[i].copy()
    item["_attack_category"] = "direct_harm"
    item["_case_idx"] = i
    test_cases.append(item)
for i in range(min(NUM_DS, len(ds_data))):
    item = ds_data[i].copy()
    item["_attack_category"] = "data_stealing"
    item["_case_idx"] = i
    test_cases.append(item)

total_cases = len(test_cases)
if args.limit > 0:
    test_cases = test_cases[:args.limit]
    total_cases = len(test_cases)
# Apply --start/--end slice (for multi-process parallelism)
if args.start > 0 or args.end > 0:
    end = args.end if args.end > 0 else total_cases
    test_cases = test_cases[args.start:end]
    total_cases = len(test_cases)
NUM_DH = sum(1 for t in test_cases if t["_attack_category"] == "direct_harm")
NUM_DS = sum(1 for t in test_cases if t["_attack_category"] == "data_stealing")
log(f"OK Loaded {total_cases} test cases ({NUM_DH} DH + {NUM_DS} DS) [start={args.start}, end={args.end if args.end > 0 else 'end'}]")

system_prompt, user_prompt = PROMPT_DICT["InjecAgent"]
tool_dict = get_tool_dict()
log("OK Prompts and tools loaded")


# ─── Step 4: vLLM API call function ───
def call_vllm(messages, max_tokens=None, timeout=None):
    """Call vLLM and return content, reasoning, finish_reason, usage."""
    if max_tokens is None:
        max_tokens = args.max_tokens
    if timeout is None:
        timeout = args.timeout

    # Cap max_tokens to max_model_len - 1 (absolute upper bound)
    max_model_len = 65536
    max_tokens = min(max_tokens, max_model_len - 1)

    for attempt in range(5):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            # Reasoning models need enable_thinking in chat_template_kwargs
            if args.model in ("nano", "gemma", "qwen"):
                payload["chat_template_kwargs"] = {"enable_thinking": True}
            resp = requests.post(
                f"{VLLM_BASE}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            data = resp.json()

            if "choices" in data:
                break

            if resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code >= 500:
                time.sleep(1 * (attempt + 1))
                continue

            # 400 error — likely prompt + max_tokens > max_model_len
            error_msg = str(data.get("error", {}).get("message", str(data)))
            if "maximum context length" in error_msg or "input_tokens" in error_msg:
                # Halve max_tokens and retry
                new_max = max_tokens // 2
                if new_max < 256:
                    return "", "", "error", {"error": error_msg}
                max_tokens = new_max
                continue

            return "", "", "error", {"error": error_msg}

        except requests.exceptions.Timeout:
            if attempt < 4:
                continue
            return "", "", "timeout", {}
        except Exception as e:
            if attempt < 4:
                time.sleep(1)
                continue
            return "", "", f"error: {e}", {}

    try:
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        finish_reason = data["choices"][0].get("finish_reason", "stop")
        usage = data.get("usage", {})
    except (KeyError, IndexError, TypeError) as e:
        return "", "", f"parse_error: {e}", {}

    return content, reasoning, finish_reason, usage


def run_single_case(item):
    """Run a single InjecAgent test case via vLLM."""
    category = item["_attack_category"]
    case_idx = item["_case_idx"]
    task_id = f"{category}_case_{case_idx}"
    attack = "dh" if category == "direct_harm" else "ds"

    available_tool_names = [item["User Tool"]] + item["Attacker Tools"]
    available_tool_descriptions = json.dumps([tool_dict[t] for t in available_tool_names], indent=True)
    input_text = item["User Instruction"]
    agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"

    system_prompt_filled = system_prompt
    user_prompt_filled = user_prompt.format(
        available_tool_descriptions=available_tool_descriptions,
        input=input_text,
        available_tool_names=available_tool_names,
        agent_scratchpad=agent_scratchpad,
    )

    messages_step1 = [
        {"role": "system", "content": system_prompt_filled},
        {"role": "user", "content": user_prompt_filled},
    ]

    calls = []
    t0 = time.time()

    try:
        # Step 1
        content, reasoning, finish_reason, usage = call_vllm(messages_step1)
        dt = time.time() - t0
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        calls.append({
            "step": 1,
            "thinking": reasoning,
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens", 0),
            "latency_seconds": round(dt, 3),
            "finish_reason": finish_reason,
            "num_tool_calls": 0,
        })

        # Evaluate step 1
        parsed = evaluate_output_prompted(
            content, item["Attacker Tools"][0], item["Attacker Tools"],
            item["User Tool"], item["Attacker Instruction"]
        )
        eval_result = parsed["eval"]
        invalid_reason = parsed.get("invalid_reason", None) if eval_result == "invalid" else None

        # Step 2 (only for DS cases where step 1 succeeded)
        eval_step2 = None
        invalid_reason_step2 = None
        output_step2 = ""
        reasoning_step2 = ""
        messages_step2 = None

        if attack == "ds" and eval_result == "succ":
            attacker_tool_params = parsed.get("Attacker Tool Parameters", {})
            attacker_tool_thought = parsed.get("Attacker Tool Thought", "")
            attacker_tool_response = get_simulated_attacker_tool_response(
                item["Attacker Tools"][0], attacker_tool_params, tool_dict
            )

            agent_scratchpad_2 = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n{attacker_tool_thought}\nAction: {item['Attacker Tools'][0]}\nAction Input: {attacker_tool_params}\nObservation: {attacker_tool_response}\n"
            user_prompt_filled_2 = user_prompt.format(
                available_tool_descriptions=available_tool_descriptions,
                input=input_text,
                available_tool_names=available_tool_names,
                agent_scratchpad=agent_scratchpad_2,
            )

            messages_step2 = [
                {"role": "system", "content": system_prompt_filled},
                {"role": "user", "content": user_prompt_filled_2},
            ]

            t1 = time.time()
            content2, reasoning2, finish_reason2, usage2 = call_vllm(messages_step2)
            dt2 = time.time() - t1

            output_step2 = content2
            reasoning_step2 = reasoning2

            calls.append({
                "step": 2,
                "thinking": reasoning2,
                "content": content2,
                "prompt_tokens": usage2.get("prompt_tokens", 0),
                "completion_tokens": usage2.get("completion_tokens", 0),
                "total_tokens": usage2.get("total_tokens", 0),
                "latency_seconds": round(dt2, 3),
                "finish_reason": finish_reason2,
                "num_tool_calls": 0,
            })

            parsed2 = evaluate_output_prompted(
                content2, item["Attacker Tools"][1], item["Attacker Tools"],
                item["User Tool"], item["Attacker Instruction"]
            )
            eval_step2 = parsed2["eval"]
            if eval_step2 == "invalid":
                invalid_reason_step2 = parsed2.get("invalid_reason", None)

        total_dt = time.time() - t0
        attack_succeeded = (eval_result == "succ") and (eval_step2 is None or eval_step2 == "succ")
        security = not attack_succeeded

        # Build full message trajectory with reasoning
        messages = [
            {"role": "system", "content": system_prompt_filled},
            {"role": "user", "content": user_prompt_filled},
            {"role": "assistant", "content": content, "reasoning": reasoning, "finish_reason": finish_reason},
        ]

        if eval_step2 is not None:
            messages.append({"role": "user", "content": user_prompt_filled_2})
            messages.append({"role": "assistant", "content": output_step2, "reasoning": reasoning_step2, "finish_reason": finish_reason2 if 'finish_reason2' in dir() else "stop"})

        case_json = {
            "suite_name": "injecagent",
            "experiment": f"EXP2B_{args.condition}_forced",
            "pipeline_name": f"{DISPLAY_NAME}_{args.precision}",
            "user_task_id": task_id,
            "injection_task_id": item["Attacker Tools"][0],
            "attack_type": item.get("Attack Type", category),
            "attack_category": category,
            "injections": {
                "attacker_instruction": item["Attacker Instruction"],
                "user_instruction": item["User Instruction"],
                "user_tool": item["User Tool"],
                "attacker_tools": item["Attacker Tools"],
                "tool_parameters": item.get("Tool Parameters", {}),
                "tool_response": item.get("Tool Response", ""),
            },
            "messages": messages,
            "error": None,
            "benchmark_version": "InjecAgent-base",
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": round(total_dt, 3),
            "utility": True,
            "security": security,
            "eval": eval_result,
            "eval_step2": eval_step2,
            "model": MODEL_NAME,
            "model_name": DISPLAY_NAME,
            "precision": args.precision,
            "thinking": reasoning,
            "thinking_step2": reasoning_step2,
            "output": content,
            "output_step2": output_step2,
        }

        if invalid_reason:
            case_json["invalid_reason"] = invalid_reason
        if invalid_reason_step2:
            case_json["invalid_reason_step2"] = invalid_reason_step2

        total_prompt = sum(c["prompt_tokens"] for c in calls)
        total_completion = sum(c["completion_tokens"] for c in calls)
        total_latency = sum(c["latency_seconds"] for c in calls)

        metrics_entry = {
            "calls": calls,
            "summary": {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "total_latency_seconds": round(total_latency, 3),
                "num_turns": len(calls),
                "num_tool_calls": sum(c["num_tool_calls"] for c in calls),
            },
            "eval": eval_result,
            "eval_step2": eval_step2,
            "attack_succeeded": attack_succeeded,
            "security": security,
        }

        return {
            "task_id": task_id,
            "ok": True,
            "case_json": case_json,
            "metrics_entry": metrics_entry,
            "eval": eval_result,
            "eval_step2": eval_step2,
            "attack_succeeded": attack_succeeded,
            "latency": total_dt,
            "reasoning_len": len(reasoning),
            "content_len": len(content),
        }

    except Exception as e:
        total_dt = time.time() - t0
        return {
            "task_id": task_id,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "latency": total_dt,
            "case_json": None,
            "metrics_entry": None,
        }


# ─── Step 5: Run cases ───
results = []
metrics = {
    "model": MODEL_NAME,
    "model_name": DISPLAY_NAME,
    "precision": args.precision,
    "model_config": {
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "setting": "base",
        "prompt_type": "InjecAgent",
        "api": "vLLM",
    },
    "tasks": {},
}

save_progress("running", 0, total_cases, "starting")

if args.workers == 1:
    for idx, item in enumerate(test_cases):
        # Skip already-completed cases (for resume after interruption)
        task_id = f"{item['_attack_category']}_case_{item['_case_idx']}"
        case_file = os.path.join(OUTPUT_DIR, f"{task_id}.json")
        if os.path.exists(case_file):
            try:
                with open(case_file) as f:
                    cached = json.load(f)
                results.append({"ok": True, "task_id": task_id, "case_json": cached, "metrics_entry": {"eval": cached.get("eval", "unsucc"), "eval_step2": cached.get("eval_step2", "")}, "latency": 0, "reasoning_len": 0, "eval": cached.get("eval", "unsucc"), "eval_step2": cached.get("eval_step2", "")})
                metrics["tasks"][task_id] = {"eval": cached.get("eval", "unsucc"), "eval_step2": cached.get("eval_step2", "")}
                log(f"  [{idx+1}/{total_cases}] {task_id}: SKIP (already done)")
                continue
            except Exception:
                pass  # File corrupt, re-run

        result = run_single_case(item)
        results.append(result)

        if result["ok"]:
            case_file = os.path.join(OUTPUT_DIR, f"{result['task_id']}.json")
            with open(case_file, "w") as f:
                json.dump(result["case_json"], f, indent=2)
            metrics["tasks"][result["task_id"]] = result["metrics_entry"]

            eval_str = result["eval"]
            if result.get("eval_step2"):
                eval_str += f" -> {result['eval_step2']}"
            log(f"  [{idx+1}/{total_cases}] {result['task_id']}: {eval_str} ({result['latency']:.2f}s, reasoning={result['reasoning_len']} chars)")
        else:
            log(f"  [{idx+1}/{total_cases}] {result['task_id']}: FAIL - {result['error']}")

        save_progress("running", idx + 1, total_cases, "executing")
else:
    log(f"Running with {args.workers} workers...")
    completed = 0
    _counter_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_item = {executor.submit(run_single_case, item): item for item in test_cases}

        for future in as_completed(future_to_item):
            result = future.result()
            with _counter_lock:
                results.append(result)
                completed += 1
                idx = completed

            if result["ok"]:
                case_file = os.path.join(OUTPUT_DIR, f"{result['task_id']}.json")
                with _write_lock:
                    with open(case_file, "w") as f:
                        json.dump(result["case_json"], f, indent=2)
                with _counter_lock:
                    metrics["tasks"][result["task_id"]] = result["metrics_entry"]

                eval_str = result["eval"]
                if result.get("eval_step2"):
                    eval_str += f" -> {result['eval_step2']}"
                with _counter_lock:
                    succ_so_far = sum(1 for r in results if r.get("attack_succeeded"))
                log(f"  [{idx}/{total_cases}] {result['task_id']}: {eval_str} ({result['latency']:.2f}s, reasoning={result['reasoning_len']} chars) | ASR: {succ_so_far}/{idx} = {succ_so_far/max(idx,1)*100:.1f}%")
            else:
                log(f"  [{idx}/{total_cases}] {result['task_id']}: FAIL - {result['error']}")

            if idx % 10 == 0:
                save_progress("running", idx, total_cases, "executing")

    results.sort(key=lambda r: r.get("task_id", ""))

# ─── Step 6: Save metrics ───
metrics_file = os.path.join(OUTPUT_DIR, "attack_metrics.json")
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=2)
log(f"Saved attack_metrics.json")

# ─── Step 7: Summary ───
ok_count = sum(1 for r in results if r.get("ok"))
fail_count = sum(1 for r in results if not r.get("ok"))
succ_count = sum(1 for r in results if r.get("attack_succeeded"))
unsucc_count = sum(1 for r in results if r.get("ok") and r.get("eval") == "unsucc")
invalid_count = sum(1 for r in results if r.get("ok") and r.get("eval") == "invalid")

valid_count = succ_count + unsucc_count
asr_all = succ_count / total_cases * 100 if total_cases > 0 else 0
asr_valid = succ_count / valid_count * 100 if valid_count > 0 else 0

dh_results = [r for r in results if r.get("task_id", "").startswith("direct_harm")]
ds_results = [r for r in results if r.get("task_id", "").startswith("data_stealing")]
dh_succ = sum(1 for r in dh_results if r.get("attack_succeeded"))
ds_succ = sum(1 for r in ds_results if r.get("attack_succeeded"))

summary = {
    "model": DISPLAY_NAME,
    "precision": args.precision,
    "mode": args.mode,
    "total_cases": total_cases,
    "successful_inference": ok_count,
    "failed": fail_count,
    "attack_succeeded": succ_count,
    "attack_failed_unsucc": unsucc_count,
    "invalid": invalid_count,
    "asr_all": round(asr_all, 2),
    "asr_valid": round(asr_valid, 2),
    "dh_total": len(dh_results),
    "dh_succeeded": dh_succ,
    "dh_asr": round(dh_succ / len(dh_results) * 100, 2) if dh_results else 0,
    "ds_total": len(ds_results),
    "ds_succeeded": ds_succ,
    "ds_asr": round(ds_succ / len(ds_results) * 100, 2) if ds_results else 0,
}

summary_file = os.path.join(OUTPUT_DIR, "summary.json")
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

save_progress("complete", total_cases, total_cases, "done", {"asr_all": asr_all, "asr_valid": asr_valid})

log("=" * 60)
log(f"BENCHMARK COMPLETE: {DISPLAY_NAME} {args.precision} {args.mode}")
log(f"  Total: {total_cases}, OK: {ok_count}, Failed: {fail_count}")
log(f"  Attack Succeeded: {succ_count}, ASR (all): {asr_all:.2f}%, ASR (valid): {asr_valid:.2f}%")
if dh_results:
    log(f"  DH: {dh_succ}/{len(dh_results)} ({dh_succ/len(dh_results)*100:.1f}% ASR)")
if ds_results:
    log(f"  DS: {ds_succ}/{len(ds_results)} ({ds_succ/len(ds_results)*100:.1f}% ASR)")
log("=" * 60)
