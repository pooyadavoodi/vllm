set -x

# export NCCL_DEBUG=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
    --port 8100 \
    --model \
    meta-llama/Meta-Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.8 \
    --max-model-len 15000 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}'
