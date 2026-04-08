#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/jyz/my_mmLLM"
LORA_PATH="/root/jyz/my_mmLLM/logs/stage2_phi4mini_20260407_025302/epoch_6"
LLM_KEY="phi4mini"
ATTR_EXP_ID="attributes_20260406_090119"
CUDA_DEVICE="3"
BATCH_SIZE="1"
MAX_NEW_TOKENS="128"

# One-click threshold sweep for ablation studies.
THRESHOLDS=(0.7 0.75 0.8 0.85 0.9 0.95)

cd "$PROJECT_ROOT"

if [[ ! -d "$LORA_PATH" ]]; then
  echo "LoRA path not found: $LORA_PATH" >&2
  exit 1
fi

run_one() {
  local thr="$1"
  local thr_tag
  thr_tag=$(python - <<PY
thr = float("${thr}")
print(int(round(thr * 100)))
PY
)

  local out_json="${PROJECT_ROOT}/processed_dataset/test_result_${LLM_KEY}_epoch_6_with_${thr_tag}.json"

  echo "============================================================"
  echo "Running threshold=${thr}"
  echo "Output: ${out_json}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  /root/anaconda3/bin/conda run -n mmllm2 --no-capture-output python evaluate_pipeline.py \
    --llm_key "${LLM_KEY}" \
    --lora_path "${LORA_PATH}" \
    --attr_exp_id "${ATTR_EXP_ID}" \
    --output_json "${out_json}" \
    --confidence_threshold "${thr}" \
    --batch_size "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}"
}

for thr in "${THRESHOLDS[@]}"; do
  run_one "${thr}"
done

echo "All threshold sweeps completed."
