from openai import AsyncOpenAI
import asyncio
from time import time

client = AsyncOpenAI(base_url="http://0.0.0.0:8000/v1/", api_key="<PARASAIL_API_KEY>")


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def get_prompt(type: str) -> str:
    if type == "very_short":
        return "What is the capital of France?"
    elif type == "short":
        return "Hughie Ferguson (2 March 1895 – 8 January 1930) was a professional footballer. He was one of Scotland's most sought-after young players before signing for Motherwell F.C. to begin his professional career. He played as a centre forward and finished as the top goalscorer in the Scottish Football League on three occasions. His 284 league goals remains a club record and, by 1925, he was the highest-scoring player in the history of the Scottish League. In 1925, Ferguson moved to Cardiff City F.C.; he was the club's top goalscorer for four consecutive seasons. He scored the winning goal in the 1927 FA Cup final and scored in the 1927 FA Charity Shield. Ferguson returned to Scotland with Dundee F.C. in 1929, but struggled to reproduce his goalscoring form. Six months after his arrival, he lost his place in the team and committed suicide. He is one of only seven men in the history of the English and Scottish Football Leagues to have scored 350 league goals."
    elif type == "med":
        return read_file("prompt_med.txt")
    elif type == "long":
        return read_file("prompt_long.txt")
    else:
        raise ValueError(f"Invalid prompt type: {type}")


def get_prompt_type(n: int) -> str:
    n = n % 4
    if n == 0:
        return "very_short"
    elif n == 1:
        return "short"
    elif n == 2:
        return "med"
    elif n == 3:
        return "long"
    raise ValueError("Invalid number")


async def run(req_id: int, prompt: str):
    print(f"Sending request: {req_id}")
    try:
        chat_completion = await client.chat.completions.create(
            model="neuralmagic/Llama-3.3-70B-Instruct-quantized.w8a8",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
    except Exception as e:
        print(f"Error in request {req_id}: {e}")
        return None
    print(f"Finished request {req_id}")
    return chat_completion


def get_bucket_info(token_lens):
    bucket_lens = [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        10000,
        50000,
        1000000,
    ]
    buckets = [[] for _ in range(len(bucket_lens))]
    for l in token_lens:
        for i, bucket_len in enumerate(bucket_lens):
            if l <= bucket_len:
                buckets[i].append(l)
                break
    print()
    for i, bucket in enumerate(buckets):
        print(f"Bucket {bucket}: {len(buckets[i])}")


async def main():
    tasks = []
    for i in range(100):
        prompt = get_prompt(get_prompt_type(i))
        tasks.append(asyncio.create_task(run(i, prompt)))

    start_time = time()
    results = await asyncio.gather(*tasks)
    duration = time() - start_time
    print(f"Time taken: {duration:.2f}s")

    in_tokens = [
        result.usage.prompt_tokens if result and result.usage else 0
        for result in results
    ]
    out_tokens = [
        result.usage.completion_tokens if result and result.usage else 0
        for result in results
    ]
    print(f"in_tokens: {in_tokens}")
    print(f"out_tokens: {out_tokens}")

    print("Input buckets:")
    get_bucket_info(in_tokens)

    print("Output buckets:")
    get_bucket_info(out_tokens)

    print(f"Input tokens per second: {sum(in_tokens) / duration:.2f}")
    print(f"Output tokens per second: {sum(out_tokens) / duration:.2f}")

    print(results[0].choices[0].message.content)
    return results


if __name__ == "__main__":
    asyncio.run(main())
