# export VLLM_LOGGING_LEVEL=DEBUG
# export VLLM_TORCH_PROFILER_DIR=/root/dev/traces
export VLLM_USE_V1=1

vllm serve \
    nm-testing/Qwen2-VL-72B-Instruct-FP8-dynamic \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt image=16 \
    --max-model-len 32768 \
    # --disable-mm-preprocessor-cache \
    # --disable-log-requests

