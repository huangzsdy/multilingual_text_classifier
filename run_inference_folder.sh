#!/usr/bin/env bash
set -u

# ================== 可调参数 ==================
GPUS="${GPUS:-0,1}"                 # 可用 GPU，逗号分隔
PER_GPU="${PER_GPU:-1}"             # 每卡最大并发任务数
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON_BIN="python3"
SCRIPT_PATH="scripts/inference.py"
# =============================================

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
GPU_IDX=0
GPU_COUNTER_DIR=$(mktemp -d)
mkdir -p logs

# ---------- GPU 轮转 ----------
next_gpu() {
    local g="${GPU_ARR[$GPU_IDX]}"
    GPU_IDX=$(( (GPU_IDX + 1) % ${#GPU_ARR[@]} ))
    echo "$g"
}

# ---------- 每卡并发控制 ----------
init_gpu_counter() {
    for g in "${GPU_ARR[@]}"; do
        echo 0 > "$GPU_COUNTER_DIR/gpu_$g"
    done
}
inc_gpu() { local f="$1"; echo $(( $(cat "$f") + 1 )) > "$f"; }
dec_gpu() { local f="$1"; echo $(( $(cat "$f") - 1 )) > "$f"; }

wait_for_slot() {
    local gpu="$1" cf="$GPU_COUNTER_DIR/gpu_$gpu"
    while true; do
        local cur=$(cat "$cf")
        if [ "$cur" -lt "$PER_GPU" ]; then
            inc_gpu "$cf"; return
        fi
        wait -n
    done
}

# ---------- 任务定义 ----------
# key => "input_folder | model_path | output_folder | log_file"
declare -A TASKS
TASKS["zh_main"]="/data/multilingual_projects/datasets/文本数据/zh \
| ../models/FacebookAI/xlm-roberta-base \
| /data/multilingual_projects/training_data/zh/all \
| logs/zh_main.log"

TASKS["en_part"]="/data/multilingual_projects/datasets/文本数据/en \
| output/results_en/checkpoint-1362 \
| /data/multilingual_projects/training_data/en/all \
| logs/en_part.log"

# ============ 执行 ============
init_gpu_counter

for key in "${!TASKS[@]}"; do
    IFS='|' read -r input_folder model_path output_folder log_file <<< "${TASKS[$key]}"
    input_folder=$(echo "$input_folder" | xargs)
    model_path=$(echo "$model_path" | xargs)
    output_folder=$(echo "$output_folder" | xargs)
    log_file=$(echo "$log_file" | xargs)

    mkdir -p "$output_folder"

    (
        gpu=$(next_gpu)
        wait_for_slot "$gpu"

        echo "[$(date '+%F %T')] [${key}] START on GPU ${gpu}" | tee -a "$log_file"

        shopt -s nullglob
        for jsonl in "$input_folder"/*.jsonl; do
            fname=$(basename "$jsonl")
            out_path="${output_folder}/${fname}"

            echo "[$(date '+%F %T')] [${key}] Processing $fname on GPU ${gpu}" | tee -a "$log_file"

            CUDA_VISIBLE_DEVICES=$gpu \
            ${PYTHON_BIN} ${SCRIPT_PATH} \
                --jsonl_files "$jsonl" \
                --batch_size ${BATCH_SIZE} \
                --output_file "$out_path" \
                --model_path "$model_path" \
                >> "$log_file" 2>&1
        done

        dec_gpu "$GPU_COUNTER_DIR/gpu_$gpu"
        echo "[$(date '+%F %T')] [${key}] DONE" | tee -a "$log_file"
    ) &
done

wait
rm -rf "$GPU_COUNTER_DIR"
echo "✅ 所有任务已完成"