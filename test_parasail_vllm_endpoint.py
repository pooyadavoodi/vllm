# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio

import requests


async def main(args):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    data = {
        "model":
        args.model,
        "messages": [{
            "role": "user",
            "content": "Write a short essay about new york city"
        }],
        "temperature":
        0.7,
        # "presence_penalty": 0.6,
        "frequency_penalty":
        0.5,
    }

    response = requests.post(args.url, headers=headers, json=data)

    print(response)
    print(response.json())
    print(response.json().get("choices")[0].get("message").get("content"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Parasail VLLM Endpoint")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help=
        "VLLM endpoint URL. Example: https://parasail-qwen25-vl-72b-instruct-28394.ohio1.sakura.saas.parasail.io/v1/chat/completions",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for VLLM endpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
