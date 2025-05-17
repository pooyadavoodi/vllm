set -v

export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=DEBUG
export HF_TOKEN="hf_"
export VLLM_USE_V1=1
# export TRITON_INTERPRET=1

# MODEL="/var/models/meta-llama/Meta-Llama-3-8B-Instruct"
# CUDA_VISIBLE_DEVICES=0,1,2,3
# export VLLM_DISAGG_PREFILL_ROLE="prefill"
# python3 -m vllm.entrypoints.openai.api_server \
#         --model $MODEL \
#         --port 8000 \
#         -tp 4



python \
    -Xfrozen_modules=off \
    -m debugpy --listen 5677 --wait-for-client \
    -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config '{ "method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4 }' \
    --tensor-parallel-size 1 \
    --enforce-eager \

# export CUDA_VISIBLE_DEVICES=3
# export VLLM_DISAGG_PREFILL_ROLE="decode"
# python3 -m vllm.entrypoints.openai.api_server \
#         --model $MODEL \
#         --port 8000 \
#         --enable-prefix-caching \
#         -tp 1 \


# sudo \
# nsys profile \
#     --gpu-metrics-device=0 \
#     -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -f true -x true  --cuda-graph-trace node \
    # /home/pooya/dev/vllm/vllm-venv/bin/vllm \
    #     serve \
    #     /var/models/meta-llama/Meta-Llama-3-8B-Instruct


#	--trace=cuda,cudnn,cublas,osrt,nvtx \
#	--force-overwrite=true \
#	--gpu-metrics-device=0 \
#        --cuda-memory-usage=true \
#        --capture-range=cudaProfilerApi \
#        --capture-range-end=stop \
#             --gpu-metrics-device=all \
#             --duration=300 \
#	--stats=true \

#    /home/pooya/dev/kashkul/a.out

# viztracer -o result-server.json \
# --log_async
