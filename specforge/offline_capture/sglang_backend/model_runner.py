import logging
import os
import time

import torch
from sglang.srt.distributed import (
    get_attn_tp_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_available_gpu_memory,
    get_bool_env_var,
    monkey_patch_p2p_access_check,
)

from .patch import (
    init_distributed_environment,
    initialize_dp_attention,
    initialize_model_parallel,
)

_is_cpu_amx_available = cpu_has_amx_support()

logger = logging.getLogger(__name__)


class SGLangRunner(ModelRunner):

    def init_torch_distributed(self):
        tic = time.perf_counter()
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.ps.tp_rank=} {self.ps.tp_size=}"
            )
            raise

        if self.device == "cuda":
            if self.server_args.elastic_ep_backend == "mooncake":
                backend = "mooncake"
                if self.server_args.mooncake_ib_device:
                    mooncake_ib_device = self.server_args.mooncake_ib_device.split(",")
                    try:
                        from mooncake import ep as mooncake_ep

                        mooncake_ep.set_device_filter(mooncake_ib_device)
                    except ImportError:
                        pass  # A warning will be raised in `init_distributed_environment`
            else:
                backend = "nccl"
        elif self.device == "xpu":
            backend = "xccl"
        elif self.device == "hpu":
            backend = "hccl"
        elif self.device == "cpu":
            backend = "gloo"
        elif self.device == "npu":
            backend = "hccl"
        elif self.device == "musa":
            backend = "mccl"

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()

        # Allow external orchestrators (e.g. trainpi) to override the distributed
        # init method.  When set to "env://", torch uses MASTER_ADDR/MASTER_PORT
        # env-vars and an externally-created TCPStore, completely avoiding port
        # conflicts with intra-host collocation.
        dist_init_method_override = envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
        if dist_init_method_override:
            dist_init_method = dist_init_method_override
        elif self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available or _is_cpu_arm64:
                    # Bind OpenMP threads to CPU cores
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.ps.tp_size)
                    torch.ops.sgl_kernel.initialize(self.ps.tp_size, self.ps.tp_rank)

                    @torch.library.register_fake("sgl_kernel::shm_allgather")
                    def _(data, dim):
                        return torch.cat([data] * self.ps.tp_size, dim=dim)

                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
                    )

            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.ps.tp_size * self.ps.pp_size,
                rank=self.ps.tp_size * self.ps.pp_rank + self.ps.tp_rank,
                local_rank=self.gpu_id,
            )
            dp_size = getattr(self.server_args, "dp_size", 1)
            attn_cp_size = getattr(self.server_args, "attn_cp_size", 1)
            moe_dp_size = getattr(self.server_args, "moe_dp_size", 1)
            initialize_model_parallel(
                tensor_model_parallel_size=self.ps.tp_size,
                pipeline_model_parallel_size=self.ps.pp_size,
                expert_model_parallel_size=self.ps.moe_ep_size,
                attention_data_parallel_size=dp_size,
                attention_context_model_parallel_size=attn_cp_size,
                moe_data_model_parallel_size=moe_dp_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
            )
            initialize_dp_attention(
                server_args=self.server_args,
                model_config=self.model_config,
            )
            if is_npu():
                register_sgl_tp_rank(self.gpu_id)

        min_per_gpu_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        self.pre_model_load_memory = min_per_gpu_memory
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attn_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.ps.tp_size > 1 and not self.is_draft_worker:
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                msg = "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                msg += f"{min_per_gpu_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                if envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get():
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

        logger.info(
            f"Init torch distributed ends. elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return min_per_gpu_memory
