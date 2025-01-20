
import copy
from typing import Dict, List, Optional

import transformers
from more_itertools import distribute
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.vllm_vlms import VLLM_VLM
from lm_eval.models.utils import (
    Collator,
    handle_stop_sequences,
    replace_placeholders,
    undistribute,
)
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.utils import eval_logger


try:
    import ray
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest  # noqa: F401
    from vllm.transformers_utils.tokenizer import get_tokenizer  # noqa: F401
except ModuleNotFoundError:
    pass


class MessLMEvalVLLM(VLLM_VLM):

	def __init__(
		self,
		pretrained: str,
		trust_remote_code: Optional[bool] = False,
		revision: Optional[str] = None,
		interleave: bool = True,
		max_images: int = 999,
		max_seq_len: int = 8192,
		gpu_indices: list = (),
		max_memory_utilization: float = 0.9,
		**kwargs,
	):
		super().__init__(pretrained, trust_remote_code, revision, interleave, max_images, **kwargs)

		self.model = LLM(
			pretrained,
			max_model_len=max_seq_len,
			trust_remote_code=True,
			tensor_parallel_size=len(gpu_indices) if len(gpu_indices) > 1 else 1,
			gpu_memory_utilization=max_memory_utilization
		)

	def _model_generate(
			self,
			requests: List[List[dict]] = None,
			generate: bool = False,
			max_tokens: int = None,
			stop: Optional[List[str]] = None,
			**kwargs,
	):
		if generate:
			kwargs = self.modify_gen_kwargs(kwargs)
			sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
		else:
			sampling_params = SamplingParams(
				temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
			)
		if self.data_parallel_size > 1:
			requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
			resps = self.model.generate(requests, sampling_params=sampling_params)

			return undistribute(resps)

		if self.lora_request is not None:
			outputs = self.model.generate(
				requests,
				sampling_params=sampling_params,
				use_tqdm=True if self.batch_size == "auto" else False,
				lora_request=self.lora_request,
			)

		else:
			outputs = self.model.generate(
				requests,
				sampling_params=sampling_params,
				use_tqdm=True if self.batch_size == "auto" else False,
			)

		return outputs
