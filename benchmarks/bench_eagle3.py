#!/usr/bin/env python3
"""
Usage:

# if you want to run benchmarks directly
# mtbench:20 means only run 20 samples in the dataset
python bench_eagle3.py \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --port 30000 \
    --config-list 1,0,0,0 1,3,1,4 \
    --benchmark-list mtbench:20 \
    --dtype bfloat16


or if you want run sglang alone.

# launch sglang
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct   \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 1 \
    --tp 1 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16

# then run benchmarks
python bench_eagle3.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --config-list 1,0,0,0 \
    --benchmark-list mtbench:80 \
    --dtype bfloat16 \
    --skip-launch-server
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import List, Optional

import requests
from benchmarker import BENCHMARKS
from sglang.lang.chat_template import ChatTemplate, register_chat_template
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import kill_process_tree, popen_launch_server
from sglang.utils import wait_for_server

register_chat_template(
    ChatTemplate(
        name="k-exaone",
        default_system_prompt="You are K-EXAONE, a large language model developed by LG AI Research in South Korea, built to serve as a helpful and reliable assistant.",
        role_prefix_and_suffix={
            "system": ("<|system|>\n", "<|endofturn|>\n"),
            "user": ("<|user|>\n", "<|endofturn|>\n"),
            "assistant": ("<|assistant|>\n<think>\n", "<|endofturn|>\n"),
        },
    )
)
register_chat_template(
    ChatTemplate(
        name="k-exaone-no-think",
        default_system_prompt="You are K-EXAONE, a large language model developed by LG AI Research in South Korea, built to serve as a helpful and reliable assistant.",
        role_prefix_and_suffix={
            "system": ("<|system|>\n", "<|endofturn|>\n"),
            "user": ("<|user|>\n", "<|endofturn|>\n"),
            "assistant": ("<|assistant|>\n<think>\n\n</think>\n\n", "<|endofturn|>\n"),
        },
    )
)


def parse_args():
    parser = argparse.ArgumentParser()
    sglang_group = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(sglang_group)

    # make the follow args a group
    benchmark_group = parser.add_argument_group("benchmark")
    benchmark_group.add_argument(
        "--skip-launch-server", action="store_true", default=False
    )
    benchmark_group.add_argument("--timeout-for-server-launch", type=int, default=600)
    benchmark_group.add_argument("--num-prompts", type=int, default=80)
    benchmark_group.add_argument("--output-dir", type=str, default="./results")
    benchmark_group.add_argument(
        "--config-list", type=str, nargs="+", default=["1,0,0,0", "1,3,1,4"]
    )
    benchmark_group.add_argument(
        "--name",
        type=str,
        default=None,
        help="name of this benchmark run, if provided, will be added to the output file name",
    )
    benchmark_group.add_argument(
        "--benchmark-list",
        type=str,
        nargs="+",
        default=[
            "mtbench:80",
            "gsm8k:200",
            "humaneval:200",
            "math500:200",
            "ceval:200",
        ],
        help=f"The list of benchmarks to run. The format is <benchmark-name>:<num-prompts>:<subset>,<subset>. We support the following benchmarks: {', '.join(BENCHMARKS.benchmarks.keys())}",
    )
    benchmark_group.add_argument(
        "--enable-multi-turn-conversation",
        action="store_true",
        default=False,
    )
    benchmark_group.add_argument(
        "--isl-list",
        type=int,
        nargs="+",
        default=[None],
        help="target input sequence length",
    )
    benchmark_group.add_argument(
        "--spec-algo",
        type=str,
        default="EAGLE3",
        help="algorithm to use for speculative decoding",
    )
    benchmark_group.add_argument(
        "--save-history",
        action="store_true",
    )
    benchmark_group.add_argument(
        "--frontend-chat-template-name",
        type=str,
        default=None,
    )

    parameter_group = parser.add_argument_group("parameter")
    parameter_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )
    parameter_group.add_argument(
        "--temperature",
        type=float,
        default=0,
    )
    parameter_group.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    parameter_group.add_argument(
        "--top-k",
        type=int,
        default=-1,
    )
    parameter_group.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
    )
    parameter_group.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def launch_sglang_server(
    server_args: ServerArgs,
    base_url: str,
    batch_size: int,
    steps: int,
    topk: int,
    num_draft_tokens: int,
    spec_algo: str,
    timeout: int,
):
    """
    This function launches the SGLang server with the given server arguments.
    """
    sglang_args: List[str] = []
    if steps > 0:
        sglang_args.extend(
            [
                "--speculative-algorithm",
                spec_algo,
                "--speculative-num-steps",
                str(steps),
                "--speculative-eagle-topk",
                str(topk),
                "--speculative-num-draft-tokens",
                str(num_draft_tokens),
            ]
        )
        if server_args.speculative_draft_model_path:
            sglang_args.extend(
                [
                    "--speculative-draft-model-path",
                    server_args.speculative_draft_model_path,
                ]
            )

    sglang_args.extend(
        [
            "--cuda-graph-max-bs",
            str(batch_size),
            "--mem-fraction-static",
            str(server_args.mem_fraction_static),
            "--tp-size",
            str(server_args.tp_size),
            "--max-running-requests",
            str(batch_size),
        ]
    )

    presented = [a for a in sys.argv[1:] if a.startswith("--")]
    exists = set(sglang_args[::2])

    default_sglang_args = ServerArgs("dummy")
    for k, v in asdict(server_args).items():
        if k == "model_path":
            continue
        argument_name = f"--{k.replace('_', '-')}"
        if (
            v != getattr(default_sglang_args, k)
            and argument_name not in exists
            and argument_name in presented
        ):
            if isinstance(v, bool):
                sglang_args.extend([argument_name])
            else:
                sglang_args.extend([argument_name, str(v)])

    process = popen_launch_server(
        server_args.model_path,
        base_url,
        timeout=timeout,
        other_args=sglang_args,
        env={
            "SGLANG_RECORD_STEP_TIME": "1",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            **os.environ,
        },
    )
    return process


def send_flush_cache_request(base_url: str):
    requests.post(base_url + "/flush_cache")


def main():
    args = parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)
    configs = [tuple(map(int, config.split(","))) for config in args.config_list]
    isls = args.isl_list

    # split the arg into list of (bench_name, num_prompts)
    benchmark_list = []
    for item in args.benchmark_list:
        splits = item.split(":")
        if len(splits) == 1:
            bench_name = splits[0]
            num_prompts = None
            subset = None
        elif len(splits) == 2:
            bench_name, num_prompts = splits
            subset = None
        elif len(splits) == 3:
            bench_name, num_prompts, subset = splits
            subset = subset.split(",")
        else:
            raise ValueError(f"Invalid benchmark list format: {item}")
        benchmark_list.append((bench_name, num_prompts, subset))
    assert len(benchmark_list) != 0, "the number of benchmark list is 0"

    results = {}
    results["model"] = server_args.speculative_draft_model_path
    results["parameter"] = dict(
        frontend_chat_template_name=args.frontend_chat_template_name,
        temeprature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        frequence_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
    )
    results_history = {}
    results_history["model"] = server_args.speculative_draft_model_path
    results_history["parameter"] = dict(
        frontend_chat_template_name=args.frontend_chat_template_name,
        temeprature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        frequence_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
    )

    def run_benchmarks(
        batch_size: int,
        steps: Optional[int],
        topk: Optional[int],
        num_draft_tokens: Optional[int],
    ):
        for benchmark_name, num_prompts, subset in benchmark_list:
            for isl in isls:
                if isl == 0:
                    isl = None
                print(
                    f"Running benchmark {benchmark_name} with {num_prompts} prompts, "
                    f"batch size {batch_size}, steps {steps}, topk {topk}, "
                    f"num_draft_tokens {num_draft_tokens}, subset {subset}, isl {isl}"
                )
                benchmarkder_cls = BENCHMARKS.get(benchmark_name)
                if args.max_new_tokens is not None:

                    def _get_max_new_tokens(self):
                        return args.max_new_tokens

                    benchmarkder_cls.get_max_new_tokens = _get_max_new_tokens

                num_prompts = int(num_prompts) if num_prompts is not None else None
                if subset is None:
                    benchmarker = benchmarkder_cls(num_samples=num_prompts)
                else:
                    benchmarker = benchmarkder_cls(
                        num_samples=num_prompts, subset=subset
                    )
                benchmarker.set_isl(isl)
                histories_list, metrics_list = benchmarker.run(
                    host=args.host,
                    port=args.port,
                    batch_size=batch_size,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    chat_template_name=args.frontend_chat_template_name,
                    reasoning_parser=args.reasoning_parser,
                )
                send_flush_cache_request(f"http://{args.host}:{args.port}")
                if benchmark_name not in results:
                    results[benchmark_name] = []
                results[benchmark_name].append(
                    dict(
                        isl=isl,
                        max_new_tokens=benchmarker.get_max_new_tokens(),
                        batch_size=batch_size,
                        steps=steps,
                        topk=topk,
                        num_draft_tokens=num_draft_tokens,
                        metrics=[asdict(metric) for metric in metrics_list],
                        num_samples=num_prompts,
                    )
                )
                if args.save_history:
                    if benchmark_name not in results_history:
                        results_history[benchmark_name] = []
                    results_history[benchmark_name].append(
                        dict(
                            isl=isl,
                            max_new_tokens=benchmarker.get_max_new_tokens(),
                            batch_size=batch_size,
                            steps=steps,
                            topk=topk,
                            num_draft_tokens=num_draft_tokens,
                            histories=histories_list,
                            num_samples=num_prompts,
                        )
                    )

    if args.skip_launch_server:
        batch_size = configs[0][0] if len(configs) > 0 else 8
        run_benchmarks(batch_size, None, None, None)
    else:
        base_url = f"http://localhost:{args.port}"
        # we itearate over each config from args
        for batch_size, steps, topk, num_draft_tokens in configs:
            process = launch_sglang_server(
                server_args,
                base_url,
                batch_size,
                steps,
                topk,
                num_draft_tokens,
                args.spec_algo,
                args.timeout_for_server_launch,
            )
            wait_for_server(base_url)
            run_benchmarks(batch_size, steps, topk, num_draft_tokens)
            kill_process_tree(process.pid)
            process.wait()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        args.output_dir,
        f"{args.name + '_' if args.name else ''}results_{timestamp}.jsonl",
    )
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {result_file}")

    if args.save_history:
        history_file = os.path.join(
            args.output_dir,
            f"{args.name + '_' if args.name else ''}histories_{timestamp}.jsonl",
        )
        with open(history_file, "w") as f:
            json.dump(results_history, f, indent=4)
        print(f"Histories saved to {history_file}")


if __name__ == "__main__":
    main()
