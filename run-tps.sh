python benchmarks/benchmark_throughput.py \
    --model nm-testing/Qwen2-VL-72B-Instruct-FP8-dynamic \
    --request-rate 16 \
    --num-prompts 128 \
    --dataset-name hf \
    --hf-output-len 1 \
    --dataset-path openbmb/RLAIF-V-Dataset \
    --hf-split train \



    --dataset-path MMMU/MMMU_Pro \
    --hf-subset vision \
    --hf-split test \


    --dataset-path openbmb/RLAIF-V-Dataset \
    --hf-split train \

    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --sharegpt-output-len 1

    --dataset-name random \
    --random-input-len 2548 \
    --random-output-len 1 \


    max_concurrency=None
    num_prompts=128, 
    request_rate=16.0
    goodput=None
    random_input_len=1024
    random_output_len=128
    random_range_ratio=1.0
    random_prefix_len=0
    hf_subset=None
    hf_split=None
    hf_output_len=None
