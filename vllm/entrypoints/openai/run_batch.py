# SPDX-License-Identifier: Apache-2.0

import asyncio
import tempfile
from http import HTTPStatus
from typing import Awaitable, Callable, List, Optional

import aiohttp
import torch
from prometheus_client import start_http_server
from tqdm import tqdm

from vllm.engine.arg_utils import AsyncEngineArgs, nullable_str
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger, logger
# yapf: disable
from vllm.entrypoints.openai.protocol import (BatchRequestInput,
                                              BatchRequestOutput,
                                              BatchResponseData,
                                              ChatCompletionResponse,
                                              EmbeddingResponse, ErrorResponse,
                                              ScoreResponse)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.entrypoints.openai.serving_score import OpenAIServingScores
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, random_uuid
from vllm.version import __version__ as VLLM_VERSION


def parse_args():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible batch runner.")
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str,
        help=
        "The path or url to a single input file. Currently supports local file "
        "paths, or the http protocol (http or https). If a URL is specified, "
        "the file should be available via HTTP GET.")
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        type=str,
        help="The path or url to a single output file. Currently supports "
        "local file paths, or web (http or https) urls. If a URL is specified,"
        " the file should be available via HTTP PUT.")
    parser.add_argument("--response-role",
                        type=nullable_str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=True`.")

    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument('--max-log-len',
                        type=int,
                        default=None,
                        help='Max number of prompt characters or prompt '
                        'ID numbers being printed in log.'
                        '\n\nDefault: Unlimited')

    parser.add_argument("--enable-metrics",
                        action="store_true",
                        help="Enable Prometheus metrics")
    parser.add_argument(
        "--url",
        type=str,
        default="0.0.0.0",
        help="URL to the Prometheus metrics server "
        "(only needed if enable-metrics is set).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the Prometheus metrics server "
        "(only needed if enable-metrics is set).",
    )
    parser.add_argument(
        "--enable-prompt-tokens-details",
        action='store_true',
        default=False,
        help="If set to True, enable prompt_tokens_details in usage.")

    return parser.parse_args()


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


class BatchProgressTracker:

    def __init__(self):
        self._total = 0
        self._pbar: Optional[tqdm] = None

    def submitted(self):
        self._total += 1

    def completed(self):
        if self._pbar:
            self._pbar.update()

    def pbar(self) -> tqdm:
        enable_tqdm = not torch.distributed.is_initialized(
        ) or torch.distributed.get_rank() == 0
        self._pbar = tqdm(total=self._total,
                          unit="req",
                          desc="Running batch",
                          mininterval=5,
                          disable=not enable_tqdm,
                          bar_format=_BAR_FORMAT)
        return self._pbar


async def read_file(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, \
                   session.get(path_or_url) as resp:
            return await resp.text()
    else:
        with open(path_or_url, encoding="utf-8") as f:
            return f.read()


async def write_local_file(output_path: str,
                           responses: List[BatchRequestOutput]) -> None:
    """
    Write the responses to a local file.
    output_path: The path to write the responses to.
    responses: The list of responses to write.
    """
    # We should make this async, but as long as run_batch runs as a
    # standalone program, blocking the event loop won't effect performance.
    with open(output_path, "w", encoding="utf-8") as f:
        for response in responses:
            print(response.model_dump_json(), file=f)


async def upload_file(output_url: str, local_data_filepath: str) -> None:
    """
    Upload a local file to a URL.
    output_url: The URL to upload the file to.
    local_data_filepath: The path to the local file to upload.
    """
    async with aiohttp.ClientSession() as session:
        with open(local_data_filepath, encoding="utf-8") as f:
            async with session.put(output_url, data=f) as response:
                if response.status != 200:
                    raise Exception(
                        f"Failed to upload file: {response.status}")


async def write_file(path_or_url: str,
                     responses: List[BatchRequestOutput]) -> None:
    """
    Write the responses to a file or upload to a URL.
    path_or_url: The path or URL to write the responses to.
    responses: The list of responses to write.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        # Write responses to a temporary file and then upload it to the URL.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as f:
            await write_local_file(f.name, responses)
            await upload_file(path_or_url, f.name)
    else:
        await write_local_file(path_or_url, responses)


def make_error_request_output(request: BatchRequestInput,
                              error_msg: str) -> BatchRequestOutput:
    batch_output = BatchRequestOutput(
        id=f"vllm-{random_uuid()}",
        custom_id=request.custom_id,
        response=BatchResponseData(
            status_code=HTTPStatus.BAD_REQUEST,
            request_id=f"vllm-batch-{random_uuid()}",
        ),
        error=error_msg,
    )
    return batch_output


async def make_async_error_request_output(
        request: BatchRequestInput, error_msg: str) -> BatchRequestOutput:
    return make_error_request_output(request, error_msg)


async def run_request(serving_engine_func: Callable,
                      request: BatchRequestInput,
                      tracker: BatchProgressTracker) -> BatchRequestOutput:
    response = await serving_engine_func(request.body)

    if isinstance(response,
                  (ChatCompletionResponse, EmbeddingResponse, ScoreResponse)):
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=BatchResponseData(
                body=response, request_id=f"vllm-batch-{random_uuid()}"),
            error=None,
        )
    elif isinstance(response, ErrorResponse):
        batch_output = BatchRequestOutput(
            id=f"vllm-{random_uuid()}",
            custom_id=request.custom_id,
            response=BatchResponseData(
                status_code=response.code,
                request_id=f"vllm-batch-{random_uuid()}"),
            error=response,
        )
    else:
        batch_output = make_error_request_output(
            request, error_msg="Request must not be sent in stream mode")

    tracker.completed()
    return batch_output


async def main(args):
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_BATCH_RUNNER)

    model_config = await engine.get_model_config()
    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    # Create the openai serving objects.
    openai_serving_models = OpenAIServingModels(
        engine_client=engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=None,
        prompt_adapters=None,
    )
    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=None,
        chat_template_content_format="auto",
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
    ) if model_config.runner_type == "generate" else None
    openai_serving_embedding = OpenAIServingEmbedding(
        engine,
        model_config,
        openai_serving_models,
        request_logger=request_logger,
        chat_template=None,
        chat_template_content_format="auto",
    ) if model_config.task == "embed" else None
    openai_serving_scores = (OpenAIServingScores(
        engine,
        model_config,
        openai_serving_models,
        request_logger=request_logger,
    ) if model_config.task == "score" else None)

    tracker = BatchProgressTracker()
    logger.info("Reading batch from %s...", args.input_file)

    # Submit all requests in the file to the engine "concurrently".
    response_futures: List[Awaitable[BatchRequestOutput]] = []
    for request_json in (await read_file(args.input_file)).strip().split("\n"):
        # Skip empty lines.
        request_json = request_json.strip()
        if not request_json:
            continue

        request = BatchRequestInput.model_validate_json(request_json)

        # Determine the type of request and run it.
        if request.url == "/v1/chat/completions":
            handler_fn = (None if openai_serving_chat is None else
                          openai_serving_chat.create_chat_completion)
            if handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg=
                        "The model does not support Chat Completions API",
                    ))
                continue

            response_futures.append(run_request(handler_fn, request, tracker))
            tracker.submitted()
        elif request.url == "/v1/embeddings":
            handler_fn = (None if openai_serving_embedding is None else
                          openai_serving_embedding.create_embedding)
            if handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg="The model does not support Embeddings API",
                    ))
                continue

            response_futures.append(run_request(handler_fn, request, tracker))
            tracker.submitted()
        elif request.url == "/v1/score":
            handler_fn = (None if openai_serving_scores is None else
                          openai_serving_scores.create_score)
            if handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg="The model does not support Scores API",
                    ))
                continue

            response_futures.append(run_request(handler_fn, request, tracker))
            tracker.submitted()
        else:
            response_futures.append(
                make_async_error_request_output(
                    request,
                    error_msg=
                    "Only /v1/chat/completions, /v1/embeddings, and /v1/score "
                    "are supported in the batch endpoint.",
                ))

    with tracker.pbar():
        responses = await asyncio.gather(*response_futures)

    logger.info("Writing batch outputs to %s...", args.output_file)
    await write_file(args.output_file, responses)


if __name__ == "__main__":
    args = parse_args()

    logger.info("vLLM batch processing API version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    # Start the Prometheus metrics server. LLMEngine uses the Prometheus client
    # to publish metrics at the /metrics endpoint.
    if args.enable_metrics:
        logger.info("Prometheus metrics enabled")
        start_http_server(port=args.port, addr=args.url)
    else:
        logger.info("Prometheus metrics disabled")

    asyncio.run(main(args))
