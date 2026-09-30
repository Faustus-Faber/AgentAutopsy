#!/bin/bash
# EXP3A v2: Reasoning Budget Constraint — focused on models with long reasoning
# Only Nanbeige and LFM have reasoning long enough to be constrained by token budgets.
# Budgets: 2000, 1000, 512, 256 tokens (aggressive enough to actually truncate reasoning)
# Other 4 models (Qwen, Gemma, Hermes, Nano) have max RL < 6K chars (~1.5K tokens) — 
#   even 512 tokens is enough for most of their cases, so budget constraint is N/A.
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

# Aggressive token budgets that will actually truncate Nanbeige's reasoning
BUDGETS=(2000 1000 512 256)

# Only Nanbeige (max RL=54K chars ~13.5K tokens) and LFM (max RL=10K chars ~2.6K tokens)
declare -A MODELS
MODELS[nanbeige]="/mnt/c/Users/T2530917/lyceum/A6000/infrastructure/start_nanbeige_fp16.sh"
MODELS[lfm]="/mnt/c/Users/T2530917/lyceum/A6000/infrastructure/start_lfm_fp16.sh"

ORDER=(nanbeige lfm)

for model in "${ORDER[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        echo ""
        echo "================================================================"
        echo "EXP3A: Running $model with max_tokens=$budget (FP16, 200 DH cases)"
        echo "================================================================"

        OUTPUT_DIR="$RESULTS_BASE/$model/tokens_$budget"
        mkdir -p "$OUTPUT_DIR"

        if [ -f "$OUTPUT_DIR/summary.json" ]; then
            echo "SKIP: $model tokens_$budget already complete"
            continue
        fi

        echo "Starting vLLM for $model..."
        bash "${MODELS[$model]}" $PORT > "$OUTPUT_DIR/vllm_server.log" 2>&1 &
        VLLM_PID=$!
        echo "vLLM PID: $VLLM_PID"

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
            echo "ERROR: vLLM failed to start"
            kill $VLLM_PID 2>/dev/null || true
            pkill -f "vllm serve" 2>/dev/null || true
            sleep 10
            continue
        fi

        sleep 5

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

        echo "Stopping vLLM..."
        kill $VLLM_PID 2>/dev/null || true
        sleep 10
        pkill -f "vllm serve" 2>/dev/null || true
        sleep 5

        echo "Done with $model tokens_$budget"
    done
done

echo ""
echo "================================================================"
echo "EXP3A v2: All models complete!"
echo "================================================================"
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
