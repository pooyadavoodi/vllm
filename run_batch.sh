set -x

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m vllm.entrypoints.openai.run_batch \
    --model nm-testing/Qwen2-VL-72B-Instruct-FP8-dynamic \
    --input-file input.jsonl \
    --output-file output.jsonl \
