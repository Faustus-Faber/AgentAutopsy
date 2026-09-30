#!/bin/bash
# EXP3A: Reasoning Budget Constraint — Master Orchestration Script
# Tests whether limiting reasoning depth (max_tokens) causally reduces ASR.
# Uses original InjecAgent runner (no prompt modification) with different max_tokens values.
# Conditions: 16000, 8000, 4000 tokens (baseline = 60000, already collected)
# Sample: 200 DH cases per model per condition (first 200 DH = most security-relevant)
set +e

source /home/t2530917/miniforge/etc/profile.d/conda.sh
conda activate toggle
export CUDA_HOME=/home/t2530917/miniforge/envs/toggle
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
export VLLM_USE_V2_MODEL_RUNNER=0
export OPENAI_API_KEY=EMPTY

PORT=8001
RUNNER="/mnt/c/Users/T2530917/lyceum/A6000/thesis/injecagent_src/run_injecagent_vllm.py"
RESULTS_BASE="/mnt/c/Users/T2530917/lyceum/A6000/thesis/EXP3A/results"

# Token budgets to test (baseline 60000 already collected)
BUDGETS=(16000 8000 4000)

# Model configs: (key, start_script)
declare -A MODELS
MODELS[qwen]="/mnt/c/Users/T2530917/lyceum/A6000/infrastructure/start_qwen_fp16.sh"
MODELS[gemma]="/mnt/c/Users/T2530917/lyceum/A6000/infrastructure/start_gemma_fp16.sh"
MODELS[nano]="/mnt/c/Users/T2530917/lyceum/A6000/infrastructure/start_nano_vllm.sh"
MODELS[nanbeige]="/mnt/c/Users/T2530917/lyceum/A6000/infrastructure/start_nanbeige_fp16.sh"
MODELS[hermes]="/mnt/c/Users/T2530917/lyceum/A6000/infrastructure/start_hermes_text_fp16.sh"
MODELS[lfm]="/mnt/c/Users/T2530917/lyceum/A6000/infrastructure/start_lfm_fp16.sh"

# Order: highest baseline ASR first (most room to observe effect)
ORDER=(qwen hermes gemma nano nanbeige lfm)

for model in "${ORDER[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        echo ""
        echo "================================================================"
        echo "EXP3A: Running $model with max_tokens=$budget (FP16, 200 DH cases)"
        echo "================================================================"

        OUTPUT_DIR="$RESULTS_BASE/$model/tokens_$budget"
        mkdir -p "$OUTPUT_DIR"

        # Skip if already complete
        if [ -f "$OUTPUT_DIR/summary.json" ]; then
            echo "SKIP: $model tokens_$budget already complete"
            continue
        fi

        # Start vLLM server
        echo "Starting vLLM for $model..."
        bash "${MODELS[$model]}" $PORT > "$OUTPUT_DIR/vllm_server.log" 2>&1 &
        VLLM_PID=$!
        echo "vLLM PID: $VLLM_PID"

        # Wait for server to be ready (up to 10 minutes)
        echo "Waiting for vLLM to be ready..."
        READY=0
        for i in $(seq 1 60); do
            if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
                echo "vLLM ready after ${i}0 seconds"
                READY=1
                break
            fi
            sleep 10
        done

        if [ $READY -eq 0 ]; then
            echo "ERROR: vLLM failed to start after 10 minutes"
            kill $VLLM_PID 2>/dev/null || true
            pkill -f "vllm serve" 2>/dev/null || true
            sleep 10
            continue
        fi

        sleep 5

        # Run InjecAgent with limited max_tokens (200 DH cases only)
        # --limit 200 takes first 200 cases which are all DH (DH loaded first)
        echo "Running InjecAgent with max_tokens=$budget for $model..."
        cd /mnt/c/Users/T2530917/lyceum/A6000/thesis/injecagent_src
        timeout 3600 python "$RUNNER" \
            --model "$model" \
            --precision fp16 \
            --output_dir "$OUTPUT_DIR" \
            --workers 16 \
            --max_tokens "$budget" \
            --limit 200 \
            > "$OUTPUT_DIR/run.log" 2>&1
        RUN_EXIT=$?

        echo "InjecAgent exit code: $RUN_EXIT"

        # Kill vLLM server
        echo "Stopping vLLM (PID $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || true
        sleep 10
        pkill -f "vllm serve" 2>/dev/null || true
        sleep 5

        echo "Done with $model tokens_$budget"
    done
done

echo ""
echo "================================================================"
echo "EXP3A: All models complete!"
echo "================================================================"
echo "Results in: $RESULTS_BASE"
for model in "${ORDER[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        SUMMARY="$RESULTS_BASE/$model/tokens_$budget/summary.json"
        if [ -f "$SUMMARY" ]; then
            echo "--- $model tokens_$budget ---"
            cat "$SUMMARY"
            echo ""
        fi
    done
done
