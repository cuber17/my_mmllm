#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/jyz/my_mmLLM"
RESULT_DIR="${PROJECT_ROOT}/processed_dataset"
LLM_KEY="phi4mini"
EPOCH="6"
HF_ENDPOINT="https://hf-mirror.com"

# One-click threshold sweep for ablation studies.
THRESHOLDS=(0.7 0.75 0.8 0.85 0.9 0.95)

cd "$PROJECT_ROOT"

processed=0

for thr in "${THRESHOLDS[@]}"; do
  thr_tag=$(python - <<PY
thr = float("${thr}")
print(int(round(thr * 100)))
PY
)

  input_file="${RESULT_DIR}/test_result_${LLM_KEY}_epoch_${EPOCH}_with_${thr_tag}.json"

  if [[ ! -f "$input_file" ]]; then
    echo "Skip missing file: ${input_file}" >&2
    continue
  fi

  input_name="$(basename "$input_file")"
  output_file="${RESULT_DIR}/benchmark_metrics_${input_name#test_result_}"

  echo "============================================================"
  echo "Input : ${input_file}"
  echo "Output: ${output_file}"
  echo "============================================================"

  /root/anaconda3/bin/conda run -n mmllm2 --no-capture-output python calculate_benchmarks.py \
    --base-dir "$PROJECT_ROOT" \
    --input-file "$input_file" \
    --output-file "$output_file" \
    --hf-endpoint "$HF_ENDPOINT"

  processed=$((processed + 1))
done

if [[ "$processed" -eq 0 ]]; then
  echo "No result files were processed from configured thresholds." >&2
  exit 1
fi

echo "All benchmark evaluations completed."
