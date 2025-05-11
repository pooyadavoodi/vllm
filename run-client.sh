set -v

# MODEL="/var/models/meta-llama/Meta-Llama-3-8B-Instruct"

export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=DEBUG

export OPENAI_API_KEY=EMPTY

# python \
        # -m lm_bench \
        # --aim_repo=. \
        # --model=/var/models/meta-llama/Meta-Llama-3-8B-Instruct \
        # --benchmark="throughput" \
        # --dataset="rag" \
        # --throughput_iterations=3 \
        # --openai_base_url="http://localhost:8000/v1" \
        # --engine=async_openai

# viztracer -o result-client.json \
# --log_async 
#        --latency_samples=5 \
#        --throughput_iterations=5 \


##########

# python -m lm_bench --aim_repo=. \
#     --model=$MODEL \
#     --benchmark="latency" \
#     --dataset="dolly" \
#     --latency_samples=5 \
#     --engines_tensor_parallel_size=2 \
#     --engine=async_vllm

###########

# python \
#     -m lm_bench \
#     --aim_repo=. \
#     --model=$MODEL \
#     --benchmark="latency" \
#     --dataset="dolly" \
#     --engines_max_output_tokens_count 1 \
#     --openai_base_url="http://localhost:8000/v1" \
#     --engine=async_openai

# python \
#         -m lm_bench \
#         --aim_repo=. \
#         --model=$MODEL \
#         --benchmark="throughput" \
#         --dataset="rag" \
#         --engines_max_output_tokens_count=1 \
#         --throughput_iterations=2 \
#         --throughput_samples=16 \
#         --openai_base_url="http://localhost:8000/v1" \
#         --engine=async_openai
