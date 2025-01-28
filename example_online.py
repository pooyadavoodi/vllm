import asyncio
import base64
import dataclasses
import io
import os
import time
import argparse
from PIL import ImageOps
import aiohttp
from datasets import load_dataset
import numpy as np
import random
import string

os.environ["VLLM_TORCH_PROFILER_DIR"] = "/root/dev/traces"

PARASAIL_API_KEY = "psk-pooyXziLMged-IO7MX20Z3kM9kLZIxtgU"

MODELS = {
    "Qwen2-VL-7B-Instruct": {
        "model_id": "cloud-qwen2-vl-7b-instruct",
        "api_key": PARASAIL_API_KEY,
    },
    "Qwen2-VL-72B-Instruct": {
        "model_id": "parasail-qwen2-vl-72b-instruct",
        "api_key": PARASAIL_API_KEY,
    },
    "Phi-3.5-vision-instruct": {
        "model_id": "cloud-microsoft-phi-3-5-vision-inst",
        "api_key": PARASAIL_API_KEY,
    },
    "Qwen2-VL-72B-Instruct-FP8": {
        "model_id": "parasail-qwen2-vl-72b-instruct-fp8",
        "api_key": PARASAIL_API_KEY,
    },
}

VLLM_MODELS = {
    "Qwen2-VL-72B-Instruct": {
        "model_id": "Qwen/Qwen2-VL-72B-Instruct",
        "api_key": "EMPTY",
    },
    "Qwen2-VL-72B-Instruct-FP8": {
        "model_id": "nm-testing/Qwen2-VL-72B-Instruct-FP8-dynamic",
        "api_key": "EMPTY",
    },
    "Llava-7B": {"model_id": "llava-hf/llava-1.5-7b-hf", "api_key": "EMPTY"},
    "Qwen2-VL-2B-Instruct": {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "api_key": "EMPTY",
    },
}

ENDPOINT_BASE_URL = "https://api.parasail.io"
ENDPOINT_URL = ENDPOINT_BASE_URL + "/chat/completions"

# Benchmark request rate (QPS). Use smaller QPS for larger models.
REQUEST_RATES = {
    "Qwen2-VL-7B-Instruct": 40,
    "Qwen2-VL-72B-Instruct": 16,
    "Phi-3.5-vision-instruct": 40,
    "Qwen2-VL-72B-Instruct-FP8": 16,
    "Llava-7B": 40,
    "Qwen2-VL-2B-Instruct": 40,
}

# Maximum number of output tokens.
MAX_TOKENS = 1

# Input images will be resized to be fit in (IMAGE_SIZE_LIMIT, IMAGE_SIZE_LIMIT).
IMAGE_SIZE_LIMIT = (1920, 1080)
# IMAGE_SIZE_LIMIT = (128, 128)
print(f"Image size: {IMAGE_SIZE_LIMIT}")

NUM_IMAGE_TOKENS = {
    "llava-hf/llava-1.5-7b-hf": 680,
    "Qwen/Qwen2-VL-2B-Instruct": 35,
    "nm-testing/Qwen2-VL-72B-Instruct-FP8-dynamic": 1975,
}


@dataclasses.dataclass
class Metrics:
    latency_s: float
    num_prompt_tokens: int
    num_completion_tokens: int


# A function that generates a random string of length n
def random_string(n: int):
    str = "IZ7N2U 2RXC\x0cAQNDCJP PO4\rLC5ZC7S7HZ97D0T Y\tP7I\nE9E\x0b\n0BGRDZN1A1N\x0c5TCGZMQH2\r49YW74PG6248  CBEYB\rUHBJZ\x0c0Y\nZ\x0cELIR74Y76UJ\rJN\x0b4Z247OL9E8ZYHK5IZKW\x0bOPUN\rEB\n4KZ\tAUNZ5 AYATH\rEAVWCBL\nXSUSBC6869K\nQUXIS0 LWW676SEEN8LAVAFDE3RK9LIKQCD4\n\x0bNGHLZCUN2T1YP\x0bN1AHX40 7OLEBK3\rURVX\r\r\x0cB6\x0c9A2RF\x0b\x0bUL414\rW\x0c7TK0LA218SAU6I\nX80B\x0cKOE3\x0cB\x0b2FPRLX9\rHBDD6WOG5PJ4Q2\nEQY8X8K7\rARCHSIY3L\tNQE\x0bV 1UFV22S0AUNEI7OI53Z1BC6AF39G\tT3VMZC8C\x0c92N\r330S1OI64F\x0cC5\n\t XNI5P1DGCV2S21SECJZ\x0c5M4ERCW\tRP7FB6\tV\x0cCYY\t4T\rYNF1RUXG86FUV\t88\r8Q\x0cZUGSGBIE\x0bGTFMQ\x0cNN12U5O7VO\x0c2E0NPFQVT\tD4ZK23K3IZ\t\tLWQ  A0\nZRK5\x0b44Q\rZEEK\tQ\tEFLN3VWWSAEB8\x0bRYS97D\nUMSWI35WU  GVHUZ22ZOPL07J\x0c7\x0bB9SUUUL5YS1XQ\x0bLOFS\tFDND\tVQLN9W PYV6QK\nAE3\x0bFL 8KT3\x0cDBZG UUWV\t6B\x0b LWGWY52\rCIRN\rUDY2N\tPVA9H0NYYOFWXA8P8I7WBJJHVLQMUC7QLB3W0L5\n\x0cD1EVQ1\nML6QDM\x0bY2D21G1BP3YGU3\n3R\nBFRM\rFDOPN6 \t\x0c9W0348IUWIQ2Y7HBFK0R2\tW\r4H17\x0c3ZLXQXI7ERMBT7P4F1GIUOHFW\rSHFG\rU8X6EU\nAEFMWX K5K\x0c9KB77VUI3AMQOBMU\x0bEVPD\n ETRL169UD2B820 \rT  Q5\x0c\tCRFX\x0cN\x0cG46\x0c8\nHUC1BTMNXD2R\nFET4SXIEO5 2FPQH5\nI7RAC25\x0cW8EBQSYT\nYDR\t06RD2ZM6E2\rKLRHPUB7 \nSF421IKYCB\tT\x0cG ZOPFBFTAOURF6GLR\x0bVRB7\nD8K\x0b I8F2BERWKO\t09NIVI\x0bTH\rX3TJ8U21V\tPHC2G54S6IJ7E\x0bXJ\x0b8\x0b3ILA\x0b0ZNA1\x0c\t\nXR2D CW\x0cH9KBBQN1OG9SGD\rYEEZ3V62TAMRMYA0913E\tU164JH63ND3\rHFF4HE86WNL2MGM\x0bJISVV4417UUJR1QNIXMZTO\tIF0N 2PV\x0cOL4VQ D\n\r\tR\rVJ7VGSL3GJBCMXUIG2\rB\x0cDQP\nWTLTZD2S56QT8S\x0cYIZVZ0J4A9E8DL24EZV29FPI14485RBSC\x0bEUM8\rZJQNVWES EMCO8NBA\r\r\x0cG9J4QBTFRN\x0c8LIUG995AR\x0cCQ32Y1ZY2A\x0bAZD0L\n1B\rHK\nFZ5XKG188ER52SSYGQ3NK\x0cAS\x0cP\t97W22RSCT28V0CV\rRE5W5E WOW\t\x0cGL\rFUH D\nARRUWJ\nBMUF9ZF\x0cUY\rR0\rBWV1\x0c69ZTS  5 10B3DTSY\x0b8E48O6\rRWWYD8U1KB5V\x0c9A\r39LX\t26Y4 GET TD5DPXU\x0b5\x0cXC87O6\rOO58CGM6DYYR56ZN\x0c\x0c1JUCN\tQ\rW\nGU\tSQDUPTD76G\x0bLYGV4B 8HMOIXP61BZVJRF4JJ3YIXAS\nHFNO 2D\tOJ1VP\rNYVP6\nV \x0bAI\x0bVA\nY7 EPQ3\rE2 D4I\x0c1MYB5\x0bAUA4PEH8BHWAKB8I14 NUHX\t1CKO7\x0b\x0b6C03ADLM9UR\nEF7HQB37K\x0bT6BZBGV\x0bWXMB\x0c4\x0cI83YJ5\t\rO\x0c 5A1\t TAO7FA3\t74TRN463Z5LL3 OSQKH0DO4WC\nSO\t\n5B\rJBFKQ\x0bBMMT812K9PXRRT\n0OH5M\t0LHQ12I\t6F1G\rV3RWNO5ESCHE\tU1OUUM2JZF064AT5YDE56ZQM2W37V \tE\rDR82WA3PFPE8IF58PFH8O4C7VD3S\nB BQDPI\tXQ\tRBW1FJO9MCY0VHSQF0 RMV1MDW1T7GWHMNUWPW0199C22HWJSY7AME\x0bRY6V\x0bS YT9FOYV 1CGHGBBVJ2NQKMLZ43B9966P\rG\x0cEDVAMCX\nVDF57EJXNLU\n\x0cME\tFGOM\r\x0cCXTOMKN\x0cPLC5TQA5S5F6\x0cG\tOL\r01E\t\x0cTV3X\rJROIC51OB55ELJG289L\n\r\tMI65C P1IC0F\r\n9U971RLVOX\x0b\x0cCR\x0cXUN 1ZWI8TXVEN\x0cOOC557UEWU8\x0c\x0c8I92ITVBTK8591TPC6CQG\n3F\rB8\r0\x0bTHUN\n1\t\tEH \rRE8DZSCDPEH3XWHYZ\n3TH0BL910JJZ\x0bSJTPPDMA4Y2CX1V4LSW3Q7\x0c6G3BDV54\x0b8JFFT0GJX42LCE\x0b8\nE9CV\x0bAO9DPOV\r3PMG\tHQ8OKBED3QP37EHLZ\r2R\r\x0bI \tKLS26OM9YO36E\tN7PB6\tBHB8\x0c8\nXHS8B0EC9K8IGR\x0cN3P3\x0cZZBEOMKUHSV7K1T3VH4\x0bVBES2XPP\t5VEW6MVMTIHZX7XF63QL0OK MDJOHDHLQVTEQSU\x0cD0MY7AYI7EOR\x0c\n\x0bGEO AI9\x0cRPY8BAIV2N1AW3MZTDKNOS\x0bPPG0P\r EUO7B1V\n\x0cANHMXF4V5J0\tTYX3T53QH\x0c2M6AZR92FRIPSCIOL0P2R\x0c \x0b276IUQ000 P43K\tVU\nGR2TXYZICE74JM3R5CKV\nHT5LYQE\r039TK P2YH XH6JTNSGSNMY9LH35LOVSDK0L5W9QA\x0bXP1AAOP3W\rP IQRXKN\rRIL2\x0b0VEWKU7QAYB98YMPU\rU9QVKGS5LE60BR H53 \r\x0bC\n\t\nWC6OTM\nA3K6\tMNFWGCNHJKB\tJS\nV8AD1W\x0bA4YYBM4VZT\x0cNT8N40QI\x0b2O4UXFBNC7HOT4UA97B3A41F36QU X52W5 6PK3MSQMHC1W1O93\rGXFKCAOF5O0\tUO7Z\nZ3UZ\x0bVU3MK9GOH\x0c5X A1OPETKBXE\x0cXN7913P7TW11VTH2\tPLGK45Y BCB9H3LXCE3OJWIGQK9BWNQ\x0cVYF 8IGTGR532\n\rWN9FIWFK\n\nM\x0cS\nX QIEL\t\x0cCOKTRCP DIK5ZESUTU WV3ZZ4\x0cP\x0bVY80\x0bDW Z4AURNRX\x0b2KQP SAEAX3N3RF\x0cG1BFLH078\x0bX63\x0c\x0c88XPX5TH5UHGI\x0b  X\x0cVV777\t59KC5YF\t7\x0bPNN\n TVQ31S\x0bC3BRZ79NADKYQ10\nXT9IY8NCALBFGE14G4VY5P2ZYKPD\x0bAP9N\r\x0b72BXEPTU5HY7R O\tAD3\x0b0W\t4RU7\x0bKAQW\x0bY9\tKG0D\x0bZ1PSBDZ8\x0c3532\x0bXEJF4LPMMTIU\x0c71CM97MBLZV3LU52\nNS86QZ\x0c67J \t148D43ZPOSU7 \x0cOR5\tPAF3G\x0cQQ\x0c\x0cT\rL899K4077CXZNG7AYK893\x0bV3N\x0c\tPJ3\x0cIT4QN1XS\x0bREGL WBH YG46HCM\x0cNDOT8MWQ\nL 9LQFLFOD0\tB6JUNL\nUXIW5RRPUD9QHO\x0cYBBSAOEBJ5VJMOWXCO9WS1RY4AT\x0c92N\x0b\x0b64G3YXU\tWKI3D\r\n7CI\x0b\rQZ6PJXG1M4YLXVKQPIU4MLFX\x0cUK754\tB7MT0V\tNWKG9EYYG\tVLMCCBBMB2 JXTLBPM0CUSW XPPQCW2\x0bL9BU\x0c\n\x0cUWZ5CYBOA\rFJ41TAT2RM932B5K4SD4A5PZTNBKBW\tQ5VS\n7VP\x0cR\x0cKP47R\x0cZM\tZLKHLM\rG8\nWSAKRMY3\x0bZ1NZQ7\x0b\x0cTUF2RXZC 70JVCEMHR\rU0\x0bDG800BIBP7Y7U1IOCXUX\tB5QF2Z\x0cM4R6QPJB\x0c\x0cRF3\n\t7MJ3CA49Y7B900EG2KXQIIH\nN \tOMN7COPQPJEJNOM9CDPQNQ\n289Y\x0bY2\tBEIN\nQ5SXP6V\nUX1F14BK\x0c 8QKJ665Q5KZT23RJ4H4N0DYD1ISDQIVNKNQHDP\x0bA\x0bYDZU06\x0b\tYA02GW8E\tNPW3Y 4HBSUEQ59\tC5IX\x0bUN0YC2WAS7E\x0cD3WUL\rI 6TG\rC3V\rI\x0cPK\nUZVI\tP8GRZ\rYDQW\t9RRI\n\tRH\x0c3G\x0cC\x0cKA88FK92RKP47N89KB L1\x0bW3Y3I9J0ZB8VKDR5NZ\r\r52GNIXLBJMEJQ7\x0bOEZUK4B\nH\t\x0bX \tRL8RVHS\t5W\n7C1YB\x0cU2V40JY\r5WHE\tII5PQB3VITX8A\tPEFIQX\r06R3ABKERP\nJGC9STU09FT9C\tFAEN9D\x0bURM\nD\x0b1YPI\r\r6FVS55W7\tBR\x0bO\n61L1\x0c\nN3RVN3ZKO\tDL1O\n\r14T\r  4K\x0c\tJ3\nMKX7QD\nDV\x0cPYOCGG4B9D\x0cGYNSQLESNF7A1RXE9J7\x0c\r \x0c5BQTS\t0DU9 7XPECCV L\x0b9ZPO0GI98SBOZ 8IXB57L9VFLKAO2FF5WJG\x0cP\x0b QJL 2\x0c\rNVW1ASPKID0N5T71CM\nKFUX07B\x0b15I FYCS\tJ\x0bOKIWCLW3\x0cXEYZJDV4AZB0\x0bV2\nAS\r4FZBCO49XZ7QF\rW4CDGA8GFS\x0c\rQHLVB0R\nA72GMBASEOJ\rZV\rF 19MT\r7QV194UIT\t3SPN99BM\x0c\t3Z96MOYN1HP1\x0c1N 4\n6J MMURLDSM\n6PP95\n1X7Z58KJLWE6H6SGA4VTMK0A99 6G87H\rSHGBR4\n\nH03W010XQKX\x0cNF0\x0b5K8EB62LD419RDGBU4W2\tGPAC1LAZI769S3VIMXYB5F7CXGJVWQ\r\t\tUDB\x0c8RH\tSXXDJC6I7DSIKQR6NH\nJ6TECWMHU\x0cO7T9CTBW8EMDKCG5JOM NVTCUCBCR1ADSMOVN\x0cW51\r\x0cRMRD8YQCMD1I98OM Y\x0cC1\tHCI6\tIYN8 V2T5GO4UKO4FD1\x0b0I3AR73V\x0b9O15HN5VB4C4P\x0cF\x0bLAK3SUI\tFJH1F\nUG\tUPR\x0c\rRGEU5R15EI QVG8\tTDB EZFPF9YV\n1QL\x0bCSMWDQB Z12MV\x0cBA5DYOPJRCGKNCS7SAZ\tV2E\tL\x0bLX\x0cNTRC9FR\t5LI\r6T5\x0b\x0bAD3IMZRW\rJO3FEKDDX\nDGWX\nU9\x0b13\x0bZ\x0b\nDX0PI3\x0cNJVEJ\rFNSVIA\rCON45\n9Y4H7\x0cM\r1S\t94U75LF6B2AL\r RBYX7B\nA4C6K\rCKYZP\x0cKBJ\t2\tLO2303OLNCE2HVF T7GO ICY0F\tSSX3UT4\r7PP835256KC\x0b WNIKXCIYU8\nBCH10\n2T1LDK\r\x0b\r765PUL \x0bBFTV\nOIH8\nECM\x0bPTHW\tN5ALEM3X1YYVQ9M19ELDX89O3\x0bKBK0YXK\nXROZUDR\x0c\r5JGT3B\x0bJ4WG3R\r5C2539EES8 PI3TNY27COR\x0b\rL B0GYUA8\x0b"
    return str[:n]


def encode_image(image):
    buf = io.BytesIO()
    ImageOps.contain(image.convert("RGB"), IMAGE_SIZE_LIMIT).save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


async def send_request(
    session,
    model_id,
    api_key,
    question,
    image,
    url,
    use_image: bool,
    use_profile: bool = False,
):
    start_time = time.perf_counter()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    image_data = (
        {
            "type": "image_url",
            "image_url": {"url": image},
        }
        if use_image
        else {
            "type": "text",
            "text": random_string(NUM_IMAGE_TOKENS[model_id]),
        }
    )

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
                    image_data,
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
            # print(f"question: {question}")
            # print(f"response: {resp["choices"][0]["message"]}")
            return Metrics(
                (time.perf_counter() - start_time) * 1000,
                resp["usage"]["prompt_tokens"],
                resp["usage"]["completion_tokens"],
            )


def print_metrics(model, metrics):
    print(f"Model {model}:")
    for field in dataclasses.fields(Metrics):
        values = [int(getattr(metric, field.name)) for metric in metrics]
        print(
            "(",
            f"{field.name:<25}: mean: {np.mean(values):.0f}, "
            f"median: {np.median(values):.0f}, ",
            f"95%-percentile {np.percentile(values, 95):.0f}, ",
            f"total: {np.sum(values):.0f}",
            ")",
        )


async def benchmark(
    model,
    num_requests: int,
    use_vllm: bool,
    use_image: bool,
    qps: int,
    use_profiler: bool = False,
):
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

        # Use the same image for all requests.
        first_data = dataset[0]
        encoded_img = encode_image(first_data["image"])

        async def requests():
            total_requests = 0
            for row in [first_data for _ in range(2048)]:
                yield (row["question"], encoded_img, url_chat)
                total_requests += 1
                if total_requests >= num_requests:
                    break
                await asyncio.sleep(
                    np.random.exponential(1.0 / qps if qps else REQUEST_RATES[model])
                )

        async def start_profiler_request():
            yield (first_data["question"], "", url_start_profile)

        async def stop_profiler_request():
            yield (first_data["question"], "", url_stop_profile)

        tasks = []

        if use_profiler:
            async for question, image, url in start_profiler_request():
                tasks.append(
                    asyncio.create_task(
                        send_request(
                            session,
                            model_id,
                            api_key,
                            question,
                            image,
                            url,
                            use_image=use_image,
                            use_profile=True,
                        )
                    )
                )
            await asyncio.gather(*tasks)
            tasks.clear()

        async for question, image, url in requests():
            tasks.append(
                asyncio.create_task(
                    send_request(
                        session,
                        model_id,
                        api_key,
                        question,
                        image,
                        url,
                        use_image=use_image,
                    )
                )
            )
        print_metrics(model, await asyncio.gather(*tasks))
        tasks.clear()

        if use_profiler:
            async for question, image, url in stop_profiler_request():
                tasks.append(
                    asyncio.create_task(
                        send_request(
                            session,
                            model_id,
                            api_key,
                            question,
                            image,
                            url,
                            use_image=use_image,
                            use_profile=True,
                        )
                    )
                )
            await asyncio.gather(*tasks)


def main(
    model: str,
    num_requests: int,
    use_vllm: bool,
    use_image: bool,
    qps: int,
    use_profiler: bool,
):
    start_time = time.perf_counter()
    asyncio.run(benchmark(model, num_requests, use_vllm, use_image, qps, use_profiler))
    duration = time.perf_counter() - start_time
    print(f"Duration: {duration*1000:.0f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name",
        choices=VLLM_MODELS.keys(),
    )
    parser.add_argument("--use-vllm", action="store_true", required=True)
    parser.add_argument("--use-profiler", action="store_true")
    parser.add_argument("--n-reqs", type=int, help="Number of requests")
    parser.add_argument("--use-image", action="store_true")
    parser.add_argument("--qps", type=int, help="QPS")

    args = parser.parse_args()

    main(
        args.model,
        args.n_reqs,
        args.use_vllm,
        args.use_image,
        args.qps,
        args.use_profiler,
    )
