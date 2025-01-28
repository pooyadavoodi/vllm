import asyncio
import base64
import dataclasses
import io
import os
import time

from PIL import ImageOps

import aiohttp
from datasets import load_dataset
import numpy as np
import openai
from urllib.request import urlretrieve

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["VLLM_TORCH_PROFILER_DIR"] = "/root/dev/traces"

PARASAIL_API_KEY = "psk-pooyXziLMged-IO7MX20Z3kM9kLZIxtgU"

MODELS = {
    "Qwen2-VL-7B-Instruct": {"model_id": "cloud-qwen2-vl-7b-instruct", "api_key": PARASAIL_API_KEY},
    "Qwen2-VL-72B-Instruct": { "model_id": "parasail-qwen2-vl-72b-instruct", "api_key": PARASAIL_API_KEY},
    "Phi-3.5-vision-instruct": { "model_id": "cloud-microsoft-phi-3-5-vision-inst", "api_key": PARASAIL_API_KEY},
    "Qwen2-VL-72B-Instruct-FP8": { "model_id": "parasail-qwen2-vl-72b-instruct-fp8", "api_key": PARASAIL_API_KEY},
}

VLLM_MODELS = {
    "Qwen2-VL-72B-Instruct": { "model_id": "Qwen/Qwen2-VL-72B-Instruct", "api_key": "EMPTY"},
    "Qwen2-VL-72B-Instruct-FP8": { "model_id": "nm-testing/Qwen2-VL-72B-Instruct-FP8-dynamic", "api_key": "EMPTY"},
    "Llava-7B": {"model_id": "llava-hf/llava-1.5-7b-hf", "api_key": "EMPTY"}
}

ENDPOINT_BASE_URL = "https://api.parasail.io"
ENDPOINT_URL = ENDPOINT_BASE_URL + "/chat/completions"

urlretrieve("https://upload.wikimedia.org/wikipedia/commons/6/6a/Mona_Lisa.jpg", "Mona_Lisa.jpg")
urlretrieve("https://upload.wikimedia.org/wikipedia/commons/8/8a/Georges_Seurat_-_Tour_Eiffel.jpg", "Tour_Eiffel.jpg")

def to_base64_url(filename):
    with open(filename, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")


client = openai.OpenAI(api_key=PARASAIL_API_KEY, base_url="https://api.parasail.io")        

# Benchmark request rate (QPS). Use smaller QPS for larger models.
REQUEST_RATES = {
    "Qwen2-VL-7B-Instruct": 40,
    "Qwen2-VL-72B-Instruct": 16,
    "Phi-3.5-vision-instruct": 40,
    "Qwen2-VL-72B-Instruct-FP8": 16,
    "Llava-7B": 40,
}

# Number of requests.
NUM_REQUESTS = 1

# Maximum number of output tokens.
MAX_TOKENS = 1

# Input images will be resized to be fit in (IMAGE_SIZE_LIMIT, IMAGE_SIZE_LIMIT).
IMAGE_SIZE_LIMIT = (1920, 1080)


@dataclasses.dataclass
class Metrics:
    latency_s: float
    num_prompt_tokens: int
    num_completion_tokens: int


def encode_image(image):
    buf = io.BytesIO()
    ImageOps.contain(image.convert("RGB"), IMAGE_SIZE_LIMIT).save(
        buf, format="JPEG"
    )
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


# Use aiohttp directly for lower client-side overhead.
# TODO: switch to use multiprocessing
async def send_request(session, model_id, api_key, question, image, url, use_profile:bool=False):
    start_time = time.perf_counter()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {"url": image},
                    # },
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
    }

    async with session.post(url=url, json=data, headers=headers) as resp:
        if resp.status != 200:
            raise Exception(f"Request failed with status {resp.status}")
            # resp = await session.post(url=url, json=data, headers=headers)
        if use_profile:
            await resp.text()
        else:
            resp = await resp.json()
            return Metrics(
                time.perf_counter() - start_time,
                resp["usage"]["prompt_tokens"],
                resp["usage"]["completion_tokens"],
            )


def print_metrics(model, metrics):
    print(f"Model {model}:")
    for field in dataclasses.fields(Metrics):
        values = [getattr(metric, field.name) for metric in metrics]
        print(
            f"{field.name:<25}: (mean: {np.mean(values):.2f}, "
            f"median: {np.median(values):.2f}, 95%-percentile {np.percentile(values, 95):.2f})"
        )


async def benchmark(model, use_vllm:bool, use_profiler:bool = False):
    api_key = VLLM_MODELS[model]["api_key"] if use_vllm else MODELS[model]["api_key"]
    model_id = VLLM_MODELS[model]["model_id"] if use_vllm else MODELS[model]["model_id"]
    url_chat = "http://localhost:8000/v1/chat/completions" if use_vllm else ENDPOINT_URL
    url_start_profile = "http://localhost:8000/start_profile"
    url_stop_profile = "http://localhost:8000/stop_profile"

    async with aiohttp.ClientSession() as session:
        # Use RLAIF-V-Dataset for benchmark
        dataset = load_dataset(
            "openbmb/RLAIF-V-Dataset",
            split="train",
            data_files=["RLAIF-V-Dataset_000.parquet"],
        ).shuffle(seed=42)

        first_data = dataset[0]

        async def requests():
            total_requests = 0
            # Use the same image for all requests.
            for row in [first_data for i in range(256)]:
                yield (row["question"], encode_image(row["image"]), url_chat)
                total_requests += 1
                if total_requests >= NUM_REQUESTS:
                    break
                await asyncio.sleep(np.random.exponential(1.0 / REQUEST_RATES[model]))

        async def start_profiler_request():
            yield (first_data["question"], "", url_start_profile)

        async def stop_profiler_request():
            yield (first_data["question"], "", url_stop_profile)


        tasks = []

        if use_profiler:
            async for question, image, url in start_profiler_request():
                tasks.append(
                    asyncio.create_task(
                        send_request(session, model_id, api_key, question, image, url, use_profile=True)
                    )
                )
            await asyncio.gather(*tasks)
            tasks.clear()

        async for question, image, url in requests():
            tasks.append(
                asyncio.create_task(
                    send_request(session, model_id, api_key, question, image, url)
                )
            )
        print_metrics(model, await asyncio.gather(*tasks))
        tasks.clear()

        if use_profiler:
            async for question, image, url in stop_profiler_request():
                tasks.append(
                    asyncio.create_task(
                        send_request(session, model_id, api_key, question, image, url, use_profile=True)
                    )
                )
            await asyncio.gather(*tasks)


# asyncio.run(benchmark("Qwen2-VL-72B-Instruct-FP8", use_vllm=False))
# asyncio.run(benchmark("Qwen2-VL-72B-Instruct-FP8", use_vllm=True, use_profiler=True))
asyncio.run(benchmark("Llava-7B", use_vllm=True, use_profiler=False))
