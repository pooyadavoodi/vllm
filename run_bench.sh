set -x

export MODEL=""

python3 benchmarks/benchmark_serving.py \
        --backend vllm \
        --model $MODEL \
        --dataset-name "random" \
        --num-prompts 50 \
        --random-input-len 20000 \
        --random-output-len 2000 \
        --port 8000 \
        --save-result \
        --result-dir "./results" \
        --request-rate 4 \
        --ignore-eos \


# python benchmarks/benchmark_serving.py \
#     --model facebook/opt-6.7b \
#     --dataset-name sharegpt \
#     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
#     --request-rate 5 \
#     --num-prompts 400 \
#     --max-concurrency 10 \


    # --dataset-name random \
    # --random-output-len 30000 \
    # --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \

    # python benchmarks/benchmark_serving.py \
    #     --backend <backend> \
    #     --model <your_model> \
    #     --dataset-name sharegpt \
    #     --dataset-path <path to dataset> \
    #     --request-rate <request_rate> \ # By default <request_rate> is inf
    #     --num-prompts <num_prompts> # By default <num_prompts> is 1000
