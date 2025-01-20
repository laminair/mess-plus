import itertools
import json
import torch

import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.api.task
import lm_eval.models
from lm_eval.caching.cache import delete_cache
from lm_eval.evaluator_utils import (
	consolidate_group_results,
	consolidate_results,
	get_sample_size,
	get_subtask_list,
	get_task_list,
	prepare_print_tasks,
	print_writeout,
	run_task_tests,
)
from lm_eval.loggers import EvaluationTracker
from lm_eval.loggers.utils import add_env_info, add_tokenizer_info, get_git_commit_hash
from lm_eval.tasks import (
	TaskManager,
	get_task_dict,
)
from lm_eval.utils import (
	eval_logger,
	handle_non_serializable,
	hash_string,
	positional_deprecated,
	simple_parse_args_string,
)

from collections import defaultdict
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
	from lm_eval.api.model import LM
	from lm_eval.api.task import Task


@positional_deprecated
def evaluate(
		lm: "LM",
		task_dict,
		limit: Optional[int] = None,
		cache_requests: bool = False,
		rewrite_requests_cache: bool = False,
		bootstrap_iters: Optional[int] = 100000,
		write_out: bool = False,
		log_samples: bool = True,
		system_instruction: Optional[str] = None,
		apply_chat_template: Union[bool, str] = False,
		fewshot_as_multiturn: bool = False,
		verbosity: str = "INFO",
		confirm_run_unsafe_code: bool = False,
):
	"""Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests.
    :param rewrite_requests_cache: bool, optional
        Rewrites all the request cache if set to `True`.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderr. Set to 0 for skipping all stderr calculations.
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: Union[bool, str]
        Specifies whether to apply a chat template to the prompt.
        - If set to True, the default chat template is applied.
        - If set to a string, applies the specified chat template by name.
        Defaults to False (no chat template applied).
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param verbosity: str
        Verbosity level for logging
    :param confirm_run_unsafe_code: bool
        Whether to confirm running tasks marked as unsafe.
    :return
        Dictionary of results
    """

	eval_logger.setLevel(getattr(logging, f"{verbosity}"))

	if apply_chat_template:
		eval_logger.warning(
			"Chat template formatting change affects loglikelihood and multiple-choice tasks. See docs/chat-template-readme.md for details."
		)

	# tracks all Instances/requests a model must generate output on.
	requests = defaultdict(list)
	# stores the amount to pad out reqs per req. type so that
	# number of fwd passes per distributed rank is equal
	padding_requests = defaultdict(int)

	# get lists of group hierarchy and each type of request
	eval_tasks = get_task_list(task_dict)
	if not log_samples:
		if not all(
				"bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys()
				for task_output in eval_tasks
		):
			raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

	# validation checks:
	# 1.are we running multimodal task <-> non-multimodal model class, or vice-versa.
	# 2.are we running code that is marked as unsafe.
	incompatible_tasks = []
	for task_output in eval_tasks:
		task: Task = task_output.task

		if getattr(lm, "MULTIMODAL", False) != getattr(task, "MULTIMODAL", False):
			incompatible_tasks.append(task_output.task_name)
		elif getattr(task, "UNSAFE_CODE", False) and not confirm_run_unsafe_code:
			raise ValueError(
				f"Attempted to run task: {task_output.task_name} which is marked as unsafe. Set confirm_run_unsafe_code=True to run this task."
			)
	if len(incompatible_tasks) > 0:
		if not getattr(lm, "MULTIMODAL", False):
			raise ValueError(
				f"Attempted to run tasks: {incompatible_tasks} which require multimodal input, but the selected model type does not currently implement this. Multimodal support is currently restricted to the ['hf-multimodal', 'vllm-vlm'] model type."
			)
		else:
			raise ValueError(
				f"Attempted to run tasks: {incompatible_tasks} which are text-only, but used a model type which only currently supports multimodal tasks."
			)
	# end validation check

	# Cache the limit arg.
	limit_arg = limit
	limits = []
	for task_output in eval_tasks:
		task: Task = task_output.task

		limit = get_sample_size(task, limit_arg)
		limits.append(limit)
		task.build_all_requests(
			limit=limit,
			rank=lm.rank,
			world_size=lm.world_size,
			cache_requests=cache_requests,
			rewrite_requests_cache=rewrite_requests_cache,
			system_instruction=system_instruction,
			apply_chat_template=bool(apply_chat_template),
			fewshot_as_multiturn=fewshot_as_multiturn,
			chat_template=getattr(lm, "apply_chat_template")
			if apply_chat_template
			else None,
			tokenizer_name=getattr(lm, "tokenizer_name", "")
			if apply_chat_template
			else "",
		)
		eval_logger.debug(
			f"Task: {task_output.task_name}; number of requests on this rank: {len(task.instances)}"
		)
		if write_out:
			print_writeout(task)
		# aggregate Instances by LM method requested to get output.
		for instance in task.instances:
			reqtype = instance.request_type
			requests[reqtype].append(instance)

	### Run LM on inputs, get all outputs ###
	# execute each type of request
	for reqtype, reqs in requests.items():
		eval_logger.info(f"Running {reqtype} requests")
		# create `K` copies of each request `req` based off `K = req.repeats`
		cloned_reqs = []
		for req in reqs:
			cloned_reqs.extend([req] * req.repeats)

		# run requests through model
		resps = getattr(lm, reqtype)(cloned_reqs)

		# put responses from model into a list of length K for each request.
		for x, req in zip(resps, cloned_reqs):
			req.resps.append(x)

		print(cloned_reqs)
