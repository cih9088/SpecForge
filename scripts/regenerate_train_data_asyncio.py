"""
This script will re-generate the dataset from target model,
which better aligns the draft model with the target model's output distribution.

Usage:
1. Set up one or more SGLang servers for the target model.

python3 -m sglang.launch_server \
	--model meta-llama/Llama-3.1-8B-Instruct \
	--mem-fraction-static 0.75 \
	--cuda-graph-max-bs 128 \
	--tp 1 \
	--trust-remote-code \
	--host 0.0.0.0 \
	--port 30000 \
	--dtype bfloat16


2. Regenerate the dataset using the `regenerate_train_data.py` script.
python scripts/regenerate_train_data.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/sharegpt_train.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen.jsonl
"""

import argparse
import asyncio
import json
import os
import random
from typing import Any, Dict, List

from openai import AsyncOpenAI
from tqdm import tqdm

def validate_regen_input(data: Any) -> str | None:
    """Return why a ShareGPT row cannot be regenerated, or ``None``."""
    if not isinstance(data, dict):
        return "Expected a JSON object"

    return validate_conversation(
        data.get("conversations"),
        error_style="regeneration",
    )

try:
    from scripts.conversation_validation import has_think_marker, validate_conversation
except ModuleNotFoundError:
    from conversation_validation import has_think_marker, validate_conversation

def set_skipped(data: Any, error: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"status": "skipped", "error": error, "data": data}
    data["status"] = "skipped"
    data["error"] = error
    return data


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Re-generate training data using sglang model server"
    )

    # model related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, required=True)
    model_group.add_argument(
        "--reasoning",
        choices=["none", "save", "disable"],
        default="none",
        help=(
            "Reasoning mode: 'none' for standard models, 'save' to store "
            "reasoning_content, or 'disable' to disable thinking via extra_body"
        ),
    )
    model_group.add_argument(
        "--is-gpt-oss",
        action="store_true",
        help="Whether the model is a GPT-OSS model",
    )

    # sampling params
    sampling_params_group = parser.add_argument_group("sampling parameters")
    sampling_params_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sglang model server",
    )
    sampling_params_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top_p",
    )
    sampling_params_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling value sent via extra_body",
    )
    sampling_params_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Mapped to presence_penalty in the OpenAI API",
    )
    sampling_params_group.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens (default: 4096)",
    )

    # optimization
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="The number of requests to send to a single server concurrently, the total number of concurrent requests is concurrency * number of server addresses",
    )

    # data related arguments
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--input-file-path", type=str, required=True, help="Path to the input file"
    )
    data_group.add_argument(
        "--output-file-path", type=str, required=True, help="Path to the output file"
    )
    data_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="The number of samples to regenerate, if not provided, all samples will be regenerated",
    )
    data_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skip already processed samples",
    )

    # sglang server
    server_group = parser.add_argument_group("sglang server")
    server_group.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        help="Server address and port for sglang model server",
    )
    return parser.parse_args()


def get_random_reasoning_effort() -> str:
    """Get a random reasoning effort level for the model with weighted probabilities."""
    # usage example: https://huggingface.co/openai/gpt-oss-20b/discussions/28
    # Reasoning effort levels with weights: LOW(4), MEDIUM(4), HIGH(2)
    reasoning_efforts = [
        "low",
        "medium",
        "high",
    ]
    weights = [4, 4, 2]
    return random.choices(reasoning_efforts, weights=weights, k=1)[0]


def compute_context_length(conversations: List[Dict[str, Any]]) -> int:
    """
    This is a rough estimate of the context length measured in untokenized
    tokens.
    """
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            # {"role": "assistant", "content": "Hi, how can I help?"}
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        length += len(text.split())
    return length


def build_query_kwargs(args, messages, max_tokens=None):
    effective_max_tokens = max_tokens if max_tokens is not None else args.max_tokens

    query_kwargs = dict(
        model=args.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.reasoning == "disable":
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    elif args.reasoning == "save":
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra_body:
        query_kwargs["extra_body"] = extra_body
    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    return query_kwargs


async def call_sglang(
    args,
    server_address: str,
    data: List[Dict[str, Any]],
    max_tokens=None,
) -> str:
    """Send a batch of prompts to sglang /v1/completions."""
    client = AsyncOpenAI(base_url=f"http://{server_address}/v1", api_key="None")

    messages = data["conversations"]
    regenerated_messages = []

    # ignore data which starts with an assistant message
    if messages[0]["role"] == "assistant":
        data["status"] = "error"
        data["error"] = "Data starts with an assistant message"
        return data

    for message in messages:
        if message["role"] == "system":
            regenerated_messages.append(message)
        elif message["role"] == "assistant":
            continue
        elif message["role"] == "user":
            regenerated_messages.append(message)

            query_kwargs = build_query_kwargs(args, regenerated_messages, max_tokens)

            try:
                resp = await client.chat.completions.create(**query_kwargs)
            except Exception as e:
                data["status"] = "error"
                data["error"] = str(e)
                return data

            if not getattr(resp, "choices", None):
                data["status"] = "error"
                data["error"] = "No completion choices returned"
                return data
            choice = resp.choices[0]

            response_text = choice.message.content
            if args.reasoning == "disable" and (
                not isinstance(response_text, str)
                or not response_text.strip()
                or has_think_marker(response_text)
            ):
                return set_skipped(
                    data,
                    "Non-reasoning assistant response is empty or contains a thinking marker",
                )
            resp_msg = {
                "role": "assistant",
                "content": response_text,
                "finish_reason": getattr(choice, "finish_reason", None),
            }
            if args.reasoning == "save":
                reasoning_content = getattr(choice.message, "reasoning_content", None)
                if reasoning_content is None:
                    model_extra = getattr(choice.message, "model_extra", None)
                    if isinstance(model_extra, dict):
                        reasoning_content = model_extra.get("reasoning_content")
                if max_tokens is None and (
                    not isinstance(response_text, str)
                    or not response_text.strip()
                    or not isinstance(reasoning_content, str)
                    or not reasoning_content.strip()
                ):
                    data["status"] = "error"
                    data["error"] = (
                        "Reasoning generation requires non-empty assistant content "
                        "and reasoning_content"
                    )
                    return data
                if max_tokens is None and (
                    has_think_marker(response_text)
                    or has_think_marker(reasoning_content)
                ):
                    return set_skipped(
                        data,
                        "Reasoning response contains a residual thinking marker",
                    )
                resp_msg["reasoning_content"] = reasoning_content
            regenerated_messages.append(resp_msg)
        else:
            data["status"] = "error"
            data["error"] = f"Invalid message role: {message['role']}"
            return data
    data["conversations"] = regenerated_messages
    data["status"] = "success"
    return data


async def main():
    # Parse command line arguments
    args = parse_arguments()

    # Validate parameters
    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if args.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")

    print(f"Configuration:")
    print(f"  Model path: {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Temperature: {args.temperature}")
    print(f"  API URL: {args.server_address}")
    print(f"  Input file: {args.input_file_path}")
    print(f"  Output file: {args.output_file_path}")
    print(f"  Resume mode: {args.resume}")
    print("-" * 50)
    total_lines = count_lines(args.input_file_path)

    error_file_path = args.output_file_path.replace(".jsonl", "_error.jsonl")
    skipped_file_path = args.output_file_path.replace(".jsonl", "_skipped.jsonl")

    processed_ids = set()
    if args.resume:
        for path in [args.output_file_path, error_file_path, skipped_file_path]:
            if os.path.exists(path):
                with open(path, "r") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            if "id" in rec:
                                processed_ids.add(rec["id"])
                        except json.JSONDecodeError:
                            continue  # skip corrupted partial lines
        if processed_ids:
            print(f"Resume mode enabled:")
            print(f"  Found {len(processed_ids)} already processed samples")
            print("-" * 50)
        if len(processed_ids) >= total_lines:
            print(f"All {total_lines} samples already processed. Nothing to do.")
            return

    # test all server addresses
    valid_server_addresses = []
    for server_address in args.server_address:
        dummy_data = dict(
            conversations=[{"role": "user", "content": "Hello, how are you?"}]
        )
        result = await call_sglang(
            args,
            server_address,
            dummy_data,
            max_tokens=1,
        )
        if result is not None and result.get("status") == "success":
            valid_server_addresses.append(server_address)
        else:
            print(f"Server {server_address} is not available")

    if len(valid_server_addresses) == 0:
        raise ValueError("No server address is available")
    print(
        f"Using {len(valid_server_addresses)} server addresses: {valid_server_addresses}"
    )
    print("-" * 50)

    # Determine file open mode based on resume flag
    file_mode = "a" if (args.resume and processed_ids) else "w"
    print(
        f"Regenerating dataset and saving the output to {args.output_file_path} and error log to {error_file_path}"
    )
    print(
        f"File open mode: {file_mode} ({'append' if file_mode == 'a' else 'overwrite'})"
    )
    print("-" * 50)
    context_token_sum = 0
    context_token_min = None
    context_token_max = 0
    success_samples = 0
    error_samples = 0
    skipped_samples = 0

    semaphore = asyncio.Semaphore(args.concurrency * len(valid_server_addresses))

    # Use a lock to protect file writes and shared counters
    write_lock = asyncio.Lock()

    output_file_handle = open(args.output_file_path, file_mode)
    error_file_handle = open(error_file_path, file_mode)
    skipped_file_handle = open(skipped_file_path, file_mode)

    pbar = tqdm(total=total_lines, desc="Processing", initial=len(processed_ids))

    async def process_item(data, server_address):
        nonlocal context_token_sum, context_token_min, context_token_max
        nonlocal success_samples, error_samples

        async with semaphore:
            regen_data = await call_sglang(args, server_address, data)

        async with write_lock:
            if regen_data["status"] == "error":
                error_file_handle.write(
                    json.dumps(regen_data, ensure_ascii=False) + "\n"
                )
                error_samples += 1
            elif regen_data["status"] == "skipped":
                skipped_file_handle.write(
                    json.dumps(regen_data, ensure_ascii=False) + "\n"
                )
                skipped_samples += 1
            else:
                ctx_len = compute_context_length(
                    regen_data.get("conversations", [])
                )
                context_token_sum += ctx_len
                if context_token_min is None:
                    context_token_min = ctx_len
                else:
                    context_token_min = min(context_token_min, ctx_len)
                context_token_max = max(context_token_max, ctx_len)

                output_file_handle.write(
                    json.dumps(regen_data, ensure_ascii=False) + "\n"
                )
                success_samples += 1
            pbar.update(1)

    # Create all tasks
    tasks = []
    start_server_index = 0

    with open(args.input_file_path, "r") as input_file:
        for line in input_file:
            if (
                args.num_samples is not None
                and len(tasks) >= args.num_samples
            ):
                break

            data = json.loads(line.strip())
            invalid_reason = validate_regen_input(data)
            if invalid_reason is not None:
                skipped_file_handle.write(
                    json.dumps(set_skipped(data, invalid_reason), ensure_ascii=False)
                    + "\n"
                )
                skipped_samples += 1
                pbar.update(1)
                continue

            if args.resume and data.get("id") in processed_ids:
                continue

            server_address = valid_server_addresses[start_server_index]
            start_server_index = (start_server_index + 1) % len(
                valid_server_addresses
            )

            tasks.append(process_item(data, server_address))

    # Run all tasks concurrently (semaphore limits actual concurrency)
    await asyncio.gather(*tasks)

    output_file_handle.close()
    error_file_handle.close()
    skipped_file_handle.close()

    pbar.close()

    print(f"\nProcessing completed!")
    if success_samples > 0:
        avg_len = context_token_sum / success_samples
        print("Context length statistics (token count over conversations):")
        print(f"Number of successful examples: {success_samples}")
        print(f"Shortest context length: {context_token_min}")
        print(f"Longest context length: {context_token_max}")
        print(f"Average context length: {avg_len:.2f}")
    else:
        print("No successful examples to compute context length statistics.")

    total_processed = success_samples + error_samples
    previously = len(processed_ids)
    if previously > 0:
        print(f"\nResume processing completed!")
        print(f"  Previously processed: {previously}")
        print(
            f"  Newly processed: {total_processed} "
            f"({success_samples} success, {error_samples} failed, "
            f"{skipped_samples} skipped)"
        )
        print(f"  Total: {previously + total_processed}")
    else:
        print(
            f"\nProcessing completed! {success_samples} samples regenerated, "
            f"{error_samples} samples failed, {skipped_samples} samples skipped."
        )


if __name__ == "__main__":
    asyncio.run(main())
