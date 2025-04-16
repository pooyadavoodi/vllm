set -x

export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=INFO

# export VLLM_TORCH_PROFILER_DIR=/root/dev/traces

# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

# export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_TOKEN="hf_"
export MODEL=""

python -m vllm.entrypoints.openai.api_server \
    --model \
    $MODEL \
    --max-model-len 91728 \
    --tensor-parallel-size 2 \
    --sequence-parallel-size 1 \
    --disable-log-requests \

    # --max-seq-len-to-capture 16 \

    # --speculative-model \
    # --speculative-draft-tensor-parallel-size \
    # 1 \
    # --num-speculative-tokens \
    # 5 \
    # --no-enable-prefix-caching \



    # neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8 \
    # --task generate \
    # --no-enable-prefix-caching \
    # --max-model-len 92016 \
    # --disable-log-requests \
    # --speculative-model \
    # neuralmagic/Llama-3.2-1B-Instruct-quantized.w8a8 \
    # --speculative-draft-tensor-parallel-size 1 \
    # --num-speculative-tokens 5 \






    # --model \
    # Gryphe/MythoMax-L2-13b \
    # --no-enable-prefix-caching \
    # --chat-template \
    # examples/template_alpaca.jinja \
    # --speculative-model \
    # 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' \
    # --speculative-draft-tensor-parallel-size 1 \
    # --num-speculative-tokens 10 \

    # --gpu-memory-utilization 0.3 \
    # --chat-template-content-format \
    # openai \



    # --limit-mm-per-prompt image=16 \
    # --max-model-len 32768 \

    # --trust-remote-code \
    # --disable-mm-preprocessor-cache \

    # --mm-processor-kwargs '{"use_fast": "True"}' \
    # --mm-processor-kwargs '{"image_processor_type":"Qwen2VLImageProcessor", "image_processor_filename": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json"}' \
    # --mm-processor-kwargs '{"image_processor_type":"Qwen2VLImageProcessor"}' \
    # --mm-processor-kwargs '{"min_pixels":3136}' \
    # --trust-remote-code \
    # --enable-prefix-caching \
    # --enable-prompt-tokens-details \


    # --block-size 1024 \
    # --enable-prefix-caching \
    # --swap-space 1 \
    # --gpu-memory-utilization 0.3 \
    # --preemption-mode swap \
    # --disable-log-requests \
    # --max-num-batched-tokens 8532 \
    # --max-model-len 8532 \
    # --max-model-len 524288 \

# vllm serve \
    # --disable-mm-preprocessor-cache \

    # nm-testing/Qwen2-VL-7B-Instruct-FP8-dynamic \
    #nm-testing/Qwen2.5-VL-72B-Instruct-FP8-Dynamic \
    # nm-testing/Qwen2-VL-7B-Instruct-FP8-dynamic \
    # nm-testing/Qwen2.5-VL-7B-Instruct-FP8-Dynamic \
    # nm-testing/Qwen2-VL-7B-Instruct-FP8-dynamic \
    # nm-testing/Qwen2-VL-7B-Instruct-FP8-dynamic \
    # Qwen/Qwen2.5-VL-7B-Instruct \
    # Qwen/Qwen2-VL-7B-Instruct \
