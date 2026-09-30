#!/usr/bin/env python3
"""InjecAgent benchmark runner with reasoning trace capture.

Usage:
  python _benchmark.py --model_id <id> --precision <fp16|fp8|nf4> \
    --mode <smoke|full> --output_dir <dir> --reasoning_parser <parser> \
    --model_name <Gemma4|Qwen3.5>

Outputs JSON matching the existing Results/ folder structure:
- Per-case JSON files with messages, thinking, utility, security, duration
- attack_metrics.json with per-call thinking, token counts, latency
- progress.json for monitoring
"""
import json
import os
import sys
import time
import argparse
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Parse args ───
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", required=True, help="vLLM model ID")
parser.add_argument("--precision", required=True, choices=["fp16", "fp8", "nf4"])
parser.add_argument("--mode", required=True, choices=["smoke", "full"])
parser.add_argument("--output_dir", required=True, help="Output directory")
parser.add_argument("--reasoning_parser", required=True, help="Reasoning parser name")
parser.add_argument("--model_name", required=True, help="Display name (Gemma4, Qwen3.5)")
parser.add_argument("--workers", type=int, default=8, help="Concurrent workers for full mode")
parser.add_argument("--max_tokens", type=int, default=40960, help="Max tokens for generation (must be < max-model-len)")
args = parser.parse_args()

# ─── Set env vars BEFORE imports ───
VLLM_PORT = int(os.environ.get("LOCAL_LLM_PORT", 8001))
VLLM_BASE = f"http://localhost:{VLLM_PORT}/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = VLLM_BASE
os.environ["PYTHONPATH"] = "/home/user1/toggle_hijacking/InjecAgent"
sys.path.insert(0, "/home/user1/toggle_hijacking/InjecAgent")

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "run.log")

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_progress(status, completed, total, phase, extra=None):
    prog = {
        "model": args.model_name,
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
    with open(PROGRESS_FILE, "w") as f:
        json.dump(prog, f, indent=2)

# ─── Step 1: Check vLLM server ───
log(f"Starting benchmark: {args.model_name} {args.precision} {args.mode}")
log(f"Output dir: {OUTPUT_DIR}")

try:
    resp = requests.get(f"{VLLM_BASE}/models", timeout=10)
    models = resp.json()
    model_id = models["data"][0]["id"]
    log(f"OK vLLM server reachable, model: {model_id}")
except Exception as e:
    log(f"FAIL Cannot reach vLLM: {e}")
    save_progress("failed", 0, 0, "server_check", {"error": str(e)})
    sys.exit(1)

# Use the model_id from the server if not specified
if args.model_id == "auto":
    args.model_id = model_id

# Set VLLM_MODEL_ID for get_simulated_attacker_tool_response (used in DS step 2)
os.environ["VLLM_MODEL_ID"] = args.model_id

# ─── Step 2: Import InjecAgent modules ───
from src.utils import get_tool_dict, get_simulated_attacker_tool_response, get_score
from src.prompts.agent_prompts import PROMPT_DICT
from src.output_parsing import evaluate_output_prompted
from src.models import MODELS

log("OK InjecAgent modules imported")

# ─── Step 3: Load test data ───
with open("data/test_cases_dh_base.json") as f:
    dh_data = json.load(f)
with open("data/test_cases_ds_base.json") as f:
    ds_data = json.load(f)

if args.mode == "smoke":
    NUM_DH = 5
    NUM_DS = 3
else:
    NUM_DH = len(dh_data)  # 510
    NUM_DS = len(ds_data)  # 544

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
log(f"OK Loaded {total_cases} test cases ({NUM_DH} DH + {NUM_DS} DS)")

# ─── Step 4: Initialize model ───
params = {
    "model_type": "GPT",
    "model_name": args.model_id,
    "setting": "base",
    "prompt_type": "InjecAgent",
    "only_first_step": False,
    "use_cache": False,
    "only_get_score": False,
}
model_class = MODELS["GPT"](params)
system_prompt, user_prompt = PROMPT_DICT["InjecAgent"]
tool_dict = get_tool_dict()
log("OK Model initialized")

# ─── Step 5: Define case runner ───
# Cache max_model_len from server to avoid repeated queries
_max_model_len_cache = None

def get_max_model_len():
    """Get max_model_len from the vLLM server."""
    global _max_model_len_cache
    if _max_model_len_cache is not None:
        return _max_model_len_cache
    try:
        resp = requests.get(f"{VLLM_BASE}/models", timeout=10)
        _max_model_len_cache = resp.json()["data"][0].get("max_model_len", 65536)
    except:
        _max_model_len_cache = 65536
    return _max_model_len_cache

def call_vllm(messages, max_tokens=None, timeout=600):
    """Call vLLM with reasoning enabled and return content, reasoning, finish_reason, usage.

    Automatically caps max_tokens so that prompt_tokens + max_tokens <= max_model_len.
    If the request fails with a 400 error, retries with a reduced max_tokens.
    """
    if max_tokens is None:
        max_tokens = args.max_tokens

    mml = get_max_model_len()

    for attempt in range(3):
        # Cap max_tokens to max_model_len - 1 (absolute upper bound)
        capped = min(max_tokens, mml - 1)

        resp = requests.post(
            f"{VLLM_BASE}/chat/completions",
            headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
            json={
                "model": args.model_id,
                "messages": messages,
                "temperature": 0,
                "max_tokens": capped,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            timeout=timeout,
        )
        data = resp.json()

        if "choices" in data:
            break

        # 400 error — prompt + max_tokens > max_model_len
        error_msg = str(data.get("error", {}).get("message", ""))
        if resp.status_code == 400 and "maximum context length" in error_msg:
            import re
            prompt_tokens = None
            # Format 1: "prompt contains at least N input tokens"
            m = re.search(r'(\d+) input tokens', error_msg)
            if m and int(m.group(1)) > 1:
                prompt_tokens = int(m.group(1))
            # Format 2: "prompt contains N characters" — estimate tokens
            if prompt_tokens is None:
                m = re.search(r'(\d+) characters', error_msg)
                if m:
                    prompt_tokens = int(m.group(1)) // 3  # rough chars-to-tokens estimate
            if prompt_tokens is not None:
                new_max = mml - prompt_tokens - 100  # 100 token safety margin
                if new_max < 1024:
                    new_max = 1024
                log(f"  Auto-capping max_tokens: {capped} -> {new_max} (prompt~{prompt_tokens} tokens, mml={mml})")
                max_tokens = new_max
                continue
        # Non-retryable error
        break

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    finish_reason = data["choices"][0]["finish_reason"]
    usage = data.get("usage", {})
    return content, reasoning, finish_reason, usage

def run_single_case(item):
    """Run a single test case. Handles 2-step DS cases."""
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
    model_input = model_class.prepare_input(system_prompt_filled, user_prompt_filled)

    calls = []
    t0 = time.time()

    try:
        # Step 1
        content, reasoning, finish_reason, usage = call_vllm(model_input)
        dt = time.time() - t0
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        calls.append({
            "thinking": reasoning,
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
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
        if attack == "ds" and eval_result == "succ":
            # Get simulated attacker tool response
            item["Attacker Tool Parameters"] = parsed_output_get(parsed, "Attacker Tool Parameters", {})
            item["Attacker Tool Thought"] = parsed_output_get(parsed, "Attacker Tool Thought", "")
            item["Attacker Tool Response"] = get_simulated_attacker_tool_response(
                item["Attacker Tools"][0], item["Attacker Tool Parameters"], tool_dict
            )

            # Build step 2 input
            agent_scratchpad_2 = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n{item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
            user_prompt_filled_2 = user_prompt.format(
                available_tool_descriptions=available_tool_descriptions,
                input=input_text,
                available_tool_names=available_tool_names,
                agent_scratchpad=agent_scratchpad_2,
            )
            model_input_2 = model_class.prepare_input(system_prompt_filled, user_prompt_filled_2)

            t1 = time.time()
            content2, reasoning2, finish_reason2, usage2 = call_vllm(model_input_2)
            dt2 = time.time() - t1

            output_step2 = content2
            reasoning_step2 = reasoning2

            calls.append({
                "thinking": reasoning2,
                "content": content2,
                "prompt_tokens": usage2.get("prompt_tokens", 0),
                "completion_tokens": usage2.get("completion_tokens", 0),
                "total_tokens": usage2.get("total_tokens", 0),
                "latency_seconds": round(dt2, 3),
                "finish_reason": finish_reason2,
                "num_tool_calls": 0,
            })

            # Evaluate step 2
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

        # Build per-case JSON (matching existing Results structure)
        messages = [
            {"role": "system", "content": system_prompt_filled},
            {"role": "user", "content": user_prompt_filled},
            {"role": "assistant", "content": content, "reasoning": reasoning},
        ]

        # For DS 2-step cases, append step 2 turn (user scratchpad + assistant response)
        if eval_step2 is not None:
            messages.append({"role": "user", "content": user_prompt_filled_2})
            messages.append({"role": "assistant", "content": output_step2, "reasoning": reasoning_step2})

        case_json = {
            "suite_name": "injecagent",
            "pipeline_name": "local",
            "user_task_id": task_id,
            "injection_task_id": item["Attacker Tools"][0],
            "attack_type": item["Attack Type"],
            "attack_category": category,
            "injections": {
                "attacker_instruction": item["Attacker Instruction"],
                "user_instruction": item["User Instruction"],
            },
            "messages": messages,
            "error": None,
            "benchmark_version": "InjecAgent-base",
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": total_dt,
            "utility": True,
            "security": security,
            "eval": eval_result,
            "model": args.model_id,
            "model_name": args.model_name,
            "precision": args.precision,
            "thinking": reasoning,
            "output": content,
        }

        if invalid_reason:
            case_json["invalid_reason"] = invalid_reason

        if eval_step2 is not None:
            case_json["eval_step2"] = eval_step2
            case_json["output_step2"] = output_step2
            case_json["thinking_step2"] = reasoning_step2
            if invalid_reason_step2:
                case_json["invalid_reason_step2"] = invalid_reason_step2

        # Build metrics entry
        total_prompt = sum(c["prompt_tokens"] for c in calls)
        total_completion = sum(c["completion_tokens"] for c in calls)
        total_tok = sum(c["total_tokens"] for c in calls)
        total_latency = sum(c["latency_seconds"] for c in calls)

        metrics_entry = {
            "calls": calls,
            "summary": {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_tok,
                "total_latency_seconds": total_latency,
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

def parsed_output_get(parsed, key, default):
    """Safely get a key from parsed output."""
    try:
        return parsed.get(key, default)
    except:
        return default

# ─── Step 6: Run cases ───
results = []
metrics = {
    "model": args.model_id,
    "model_name": args.model_name,
    "model_config": {
        "temperature": 0.0,
        "dtype": "bfloat16",
        "max_model_len": get_max_model_len(),
        "enable_thinking": True,
        "reasoning_parser": args.reasoning_parser,
        "precision": args.precision,
        "setting": "base",
        "prompt_type": "InjecAgent",
    },
    "tasks": {},
}

save_progress("running", 0, total_cases, "starting")

if args.mode == "smoke":
    # Sequential for smoke test
    for idx, item in enumerate(test_cases):
        result = run_single_case(item)
        results.append(result)

        if result["ok"]:
            # Save per-case JSON
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
    # Parallel for full benchmark
    log(f"Running with {args.workers} workers...")
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_item = {executor.submit(run_single_case, item): item for item in test_cases}

        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)
            completed += 1

            if result["ok"]:
                # Save per-case JSON
                case_file = os.path.join(OUTPUT_DIR, f"{result['task_id']}.json")
                with open(case_file, "w") as f:
                    json.dump(result["case_json"], f, indent=2)
                metrics["tasks"][result["task_id"]] = result["metrics_entry"]

                if completed % 50 == 0 or completed == total_cases:
                    eval_str = result["eval"]
                    succ_so_far = sum(1 for r in results if r.get("attack_succeeded"))
                    log(f"  [{completed}/{total_cases}] {result['task_id']}: {eval_str} | ASR so far: {succ_so_far}/{completed} = {succ_so_far/max(completed,1)*100:.1f}%")
            else:
                log(f"  [{completed}/{total_cases}] {result['task_id']}: FAIL - {result['error']}")

            if completed % 50 == 0:
                save_progress("running", completed, total_cases, "executing")

    # Sort results by task_id for consistent output
    results.sort(key=lambda r: r.get("task_id", ""))

# ─── Step 7: Save metrics JSON ───
metrics_file = os.path.join(OUTPUT_DIR, "attack_metrics.json")
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=2)
log(f"Saved attack_metrics.json")

# ─── Step 8: Summary ───
ok_count = sum(1 for r in results if r.get("ok"))
fail_count = sum(1 for r in results if not r.get("ok"))
succ_count = sum(1 for r in results if r.get("attack_succeeded"))
unsucc_count = sum(1 for r in results if r.get("ok") and r.get("eval") == "unsucc")
invalid_count = sum(1 for r in results if r.get("ok") and r.get("eval") == "invalid")

# ASR calculations
valid_count = succ_count + unsucc_count
asr_all = succ_count / total_cases * 100 if total_cases > 0 else 0
asr_valid = succ_count / valid_count * 100 if valid_count > 0 else 0

# By category
dh_results = [r for r in results if r.get("task_id", "").startswith("direct_harm")]
ds_results = [r for r in results if r.get("task_id", "").startswith("data_stealing")]
dh_succ = sum(1 for r in dh_results if r.get("attack_succeeded"))
ds_succ = sum(1 for r in ds_results if r.get("attack_succeeded"))

summary = {
    "model": args.model_name,
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

log("=" * 60)
log(f"BENCHMARK COMPLETE: {args.model_name} {args.precision} {args.mode}")
log("=" * 60)
log(f"Total cases: {total_cases}")
log(f"Successful inference: {ok_count}/{total_cases}")
log(f"Attack succeeded: {succ_count}")
log(f"Attack failed (unsucc): {unsucc_count}")
log(f"Invalid: {invalid_count}")
log(f"ASR (all): {asr_all:.1f}%")
log(f"ASR (valid): {asr_valid:.1f}%")
log(f"DH ASR: {summary['dh_asr']:.1f}% ({dh_succ}/{len(dh_results)})")
log(f"DS ASR: {summary['ds_asr']:.1f}% ({ds_succ}/{len(ds_results)})")
log(f"Results saved to: {OUTPUT_DIR}")

# Determine pass/fail for smoke test
if args.mode == "smoke":
    if ok_count == total_cases:
        save_progress("smoke_pass", ok_count, total_cases, "complete", summary)
        log("SMOKE TEST: PASS")
        sys.exit(0)
    else:
        save_progress("smoke_fail", ok_count, total_cases, "complete", summary)
        log(f"SMOKE TEST: FAIL ({fail_count} failures)")
        sys.exit(1)
else:
    save_progress("complete", ok_count, total_cases, "complete", summary)
    log("FULL BENCHMARK: COMPLETE")
    sys.exit(0)
