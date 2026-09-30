#!/bin/bash
# EXP3A v3: Reasoning Budget Constraint — Nanbeige only, full 1054 cases
# Nanbeige is the only model with (a) long enough reasoning to constrain AND (b) non-zero ASR.
# Budgets: 2000, 1000, 512, 256 tokens
# Full 1054 cases (510 DH + 544 DS) to capture the 23 successful DS attacks.
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
RESULTS_BASE="/mnt/c/Users/T2530917/lyceum/A6000/thesis/EXP3A/results/nanbeige"

BUDGETS=(2000 1000 512 256)
START_SCRIPT="/mnt/c/Users/T2530917/lyceum/A6000/infrastructure/start_nanbeige_fp16.sh"

for budget in "${BUDGETS[@]}"; do
    echo ""
    echo "================================================================"
    echo "EXP3A: Running nanbeige with max_tokens=$budget (FP16, full 1054 cases)"
    echo "================================================================"

    OUTPUT_DIR="$RESULTS_BASE/tokens_$budget"
    mkdir -p "$OUTPUT_DIR"

    if [ -f "$OUTPUT_DIR/summary.json" ]; then
        echo "SKIP: nanbeige tokens_$budget already complete"
        continue
    fi

    echo "Starting vLLM for nanbeige..."
    bash "$START_SCRIPT" $PORT > "$OUTPUT_DIR/vllm_server.log" 2>&1 &
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

    echo "Running InjecAgent with max_tokens=$budget for nanbeige..."
    cd /mnt/c/Users/T2530917/lyceum/A6000/thesis/injecagent_src
    timeout 7200 python "$RUNNER" \
        --model nanbeige \
        --precision fp16 \
        --output_dir "$OUTPUT_DIR" \
        --workers 16 \
        --max_tokens "$budget" \
        > "$OUTPUT_DIR/run.log" 2>&1
    RUN_EXIT=$?

    echo "InjecAgent exit code: $RUN_EXIT"

    echo "Stopping vLLM..."
    kill $VLLM_PID 2>/dev/null || true
    sleep 10
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5

    echo "Done with nanbeige tokens_$budget"
done

echo ""
echo "================================================================"
echo "EXP3A v3: Nanbeige complete!"
echo "================================================================"
for budget in "${BUDGETS[@]}"; do
    SUMMARY="$RESULTS_BASE/tokens_$budget/summary.json"
    if [ -f "$SUMMARY" ]; then
        echo "--- nanbeige tokens_$budget ---"
        cat "$SUMMARY"
        echo ""
    fi
done
