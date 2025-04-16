set -x

export CUDA_VISIBLE_DEVICES=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_TOKEN="hf_"
export NCCL_DEBUG=WARN

python -m vllm.entrypoints.openai.api_server \
    --port 8200 \
    --model \
    meta-llama/Meta-Llama-3.1-8B-Instruct \
    --gpu-memory-utilization 0.8 \
    --max-model-len 15000 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}'
