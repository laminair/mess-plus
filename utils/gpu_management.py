import gc
import contextlib
import torch

from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

def reset_vllm_gpu_environment(model):
    destroy_model_parallel()
    destroy_distributed_environment()
    del model.llm_engine.model_executor
    del model
    gc.collect()
    torch.cuda.empty_cache()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    torch.cuda.synchronize()