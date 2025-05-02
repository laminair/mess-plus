import argparse
import contextlib
import gc
import json
import logging
import os

import datasets
import pandas as pd
import torch
import numpy as np

import wandb
import yaml

from numpy.random import binomial
from pathlib import Path
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

from lm_eval.utils import hash_string, handle_non_serializable
from lm_eval.evaluator_utils import (
    get_sample_size,
    get_task_list,
    print_writeout,
    consolidate_results,
    consolidate_group_results,
    get_subtask_list,
    prepare_print_tasks
)
from lm_eval.tasks import Task, TaskManager, get_task_dict
from lm_eval.api.task import Instance

from utils import is_nested_list
from utils.data_capturing import StreamingDataProcessor, SampleGenerator
from utils.mess_lm_eval_harness.vllm_v2 import MessLMEvalVLLM
from classifier.model import ContinualMultilabelBERTClassifier
from classifier.dataset import BertPandasDataset
from classifier.score_estimation import RoutingScoreEstimator

from zeus.monitor import ZeusMonitor

from collections import defaultdict
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'mess_plus.log'),
        logging.StreamHandler()
    ]
)

NUM_GPUS = torch.cuda.device_count()
PROJECT_ROOT_PATH = Path(__file__).parent.resolve()

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision('high')

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


class MessPlusAutomaticModelSelector(object):

    def __init__(self, config_file_path: str, project_name: str, wandb_entity: str = None):
        self.config = yaml.safe_load(open(config_file_path, "r"))
        self.lm_eval_config = self.config["lm_eval"]
        self.algorithm_config = self.config["algorithm"]
        self.wandb_project_name = project_name
        self.wandb_entity = wandb_entity

        self.dataset = None
        self.input_column_name = None
        self.expected_response_column_name = None

        self.__warm_up_inference_models()

        # Classifier model
        self.classifier_config = self.config["classifier_model"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_history = pd.DataFrame()
        self.class_weights = torch.ones((1, 3))

        # Loggers
        # When using Zeus, you must disable RAPL CPU monitoring as this will cause the program to fail.
        # Change "True" to "False" in file venv/lib/python3.12/site-packages/zeus/device/cpu/rapl.py (l. 137)
        self.wandb_run = None

        self.measurements = {data["category"]: [] for data in self.config["model_zoo"].values()}
        self.nll_scores = {data["category"]: [] for data in self.config["model_zoo"].values()}
        self.greedy_output = {data["category"]: [] for data in self.config["model_zoo"].values()}
        self.predictions = {data["category"]: [] for data in self.config["model_zoo"].values()}
        self.ground_truths = {data["category"]: [] for data in self.config["model_zoo"].values()}
        self.labels = {data["category"]: [] for data in self.config["model_zoo"].values()}

        self.energy_monitor = ZeusMonitor(gpu_indices=[i for i in range(NUM_GPUS)], approx_instant_energy=True)
        self.num_exploration_steps = []
        self.score_list = []
        self.model_chosen_list = []
        self.classifier_running_train_loss = 0.0
        self.classifier_train_steps = 0
        self.classifier_running_val_loss = 0.0
        self.classifier_val_steps = 0
        self.algorithm_correct_choices = 0

        # Algorithm config
        # Q is a virtual queue, i.e., we only keep the sum of all violations, no history.
        self.Q = 0.0

        # Classifier Score Estimation Function
        self.scoring_engine = RoutingScoreEstimator()

        # LM Eval Config
        task_manager = TaskManager(verbosity="INFO")

        self.limit_num_samples = self.lm_eval_config["limit_num_samples"]
        self.limits = []

        self.task_dict = get_task_dict(
            self.lm_eval_config["benchmarks"],
            task_manager
        )

        self.task_dict = self.__adjust_config(
            self.task_dict,
            gen_kwargs=self.lm_eval_config["gen_kwargs"] if "gen_kwargs" in self.config.keys() else None,
            predict_only=self.lm_eval_config["predict_only"] if "predict_only" in self.lm_eval_config.keys() else False,
            num_fewshot=self.lm_eval_config["num_fewshot"] if "num_fewshot" in self.lm_eval_config.keys() else 0,
            fewshot_random_seed=self.config["seed"]
        )

        print(self.task_dict)

        # for k, v in self.task_dict.items():
        #     self.task_dict[k]._config.__dict__.update({"repeats": self.lm_eval_config["num_repeats"]})

        self.eval_tasks = get_task_list(self.task_dict)

        # Config to capture the inference outputs for classifier validation
        self.data_writer = StreamingDataProcessor(
            save_path=f"{PROJECT_ROOT_PATH}/data/inference_outputs",
            file_prefix="inference_data_",
            save_frequency=100
        )

        self.sample_generator = SampleGenerator()

        # Warnings and Info messages at the start
        if self.classifier_config["generate_training_dataset"]:
            logger.warning(
                f"You have enabled 'generate_training_dataset'. "
                f"This will generate a dataset to train the routing classifier and will NOT run the MESS+ algorithm!"
            )

    def launch(
        self,
        apply_chat_template: bool = False,
        log_samples: bool = False
    ):

        if apply_chat_template:
            logger.warning(
                "Chat template formatting change affects loglikelihood and multiple-choice tasks. See docs/chat-template-readme.md for details."
            )
    
        # get lists of group hierarchy and each type of request
        if not log_samples:
            if not all(
                    "bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys()
                    for task_output in self.eval_tasks
            ):
                raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

        limit_arg = self.limit_num_samples
        for task_output in self.eval_tasks:
            print(task_output)
            # This is where plugin the MESS+ custom routine.
            self.run_benchmark(task_output, limit_arg)

        logger.info(f"Running the following eval tasks: {self.eval_tasks}")
        results, samples, configs, versions, num_fewshot, higher_is_better, = consolidate_results(self.eval_tasks)
        print(results)

        ### Calculate group metrics ###
        show_group_table = False
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(
                results, versions, self.task_dict
            )

        results_agg, group_agg = prepare_print_tasks(self.task_dict, results)
        subtask_list = get_subtask_list(self.task_dict)

        # collect all higher_is_better values for metrics
        # in the group's subtasks.
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if len(task_list) != 0:  # subtask list will list "task_name": [] for solo tasks
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better.keys():
                            _higher_is_better[m] = h

                        if m in _higher_is_better and _higher_is_better[m] is not None and _higher_is_better[m] != h:
                            logger.warning(f"Higher_is_better values for metric {m} in group {group} are not consistent. Defaulting to None.")
                            _higher_is_better[m] = None

                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **(
                {"groups": dict(group_agg.items())}
                if (bool(group_agg) & show_group_table)
                else {}
            ),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),
                        len(task_output.task.eval_docs),
                    ),
                }
                for task_output, limit in zip(self.eval_tasks, self.limits)
            },
        }
        if log_samples:
            results_dict["samples"] = dict(samples)

        return results_dict

    def run_benchmark(
        self,
        task_output,
        limit_arg,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        write_out: bool = False,
        log_samples: bool = False,
        bootstrap_iters: Optional[int] = 100000
    ):
        task: Task = task_output.task
        smallest_model_category, smallest_model_instance = next(iter(self.vllm_models.items()))
        smallest_model_instance = smallest_model_instance["vllm_eval_instance"]

        limit = get_sample_size(task, limit_arg)
        self.limits.append(limit)
        task.build_all_requests(
            limit=limit,
            rank=0,
            world_size=1,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=bool(apply_chat_template),
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(smallest_model_instance, "apply_chat_template") if apply_chat_template else None,
            tokenizer_name=getattr(smallest_model_instance, "tokenizer_name", "") if apply_chat_template else "",
        )
        logger.debug(
            f"Task: {task_output.task_name}; number of requests on this rank: {len(task.instances)}"
        )
        
        if write_out:
            print_writeout(task)
            
        # aggregate Instances by LM method requested to get output.
        requests = defaultdict(list)
        unique_doc_ids = {}

        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

            if reqtype in unique_doc_ids.keys():
                unique_doc_ids[reqtype].add(instance.doc_id)
            else:
                unique_doc_ids[reqtype] = {instance.doc_id}

        ### Run LM on inputs, get all outputs ###
        # execute each type of request
        instances_by_doc_id = defaultdict(list)
        for reqtype, reqs in requests.items():
            logger.info(f"Processing {reqtype} requests for {task.task_name} dataset with a total of {len(requests[reqtype])} samples.")
            # We modify this part to bundle all document ids.
            benchmark_documents_by_id = {}

            # create `K` copies of each request `req` based off `K = req.repeats`
            # cloned_reqs = []
            num_requests = 0
            for req in reqs:
                if req.doc_id in benchmark_documents_by_id.keys():
                    benchmark_documents_by_id[req.doc_id] += [req] * req.repeats
                else:
                    benchmark_documents_by_id[req.doc_id] = [req] * req.repeats

                num_requests += 1

            logger.info(f"Dataset replication factor: {num_requests / len(unique_doc_ids[reqtype])}")
            with wandb.init(
                project=self.wandb_project_name,
                name=self.make_wandb_run_name(task=task),
                entity=self.wandb_entity,
                config=self.config
            ) as run:
                self.wandb_run = run
                self.wandb_run.summary["V"] = self.algorithm_config["V"]
                self.wandb_run.summary["alpha"] = self.algorithm_config["alpha"]
                self.wandb_run.summary["c"] = self.algorithm_config["c"]

                # We initialize one classifier model for every benchmark
                self.__warmup_classifier_model()
                model_categories = [i for i in self.vllm_models.keys()]
                for timestamp, (doc_id, request_list) in enumerate(benchmark_documents_by_id.items()):
                    x_t = self.__sample_from_bernoulli(c=self.algorithm_config["c"], timestamp=timestamp)

                    if x_t == 1 or self.classifier_config["generate_training_dataset"]:
                        logger.info(f"Running request #{timestamp} - Training")
                        updated_requests, result_scores, training_sample = self.__get_training_sample(
                            request_list=request_list,
                            task=task,
                            doc_id=doc_id
                        )

                        if self.classifier_config["write_training_dataset_to_disk"]:
                            num_samples_saved = self.data_writer.process_row(sample=training_sample, benchmark_name=task.task_name)
                            logger.debug(f"Saved {num_samples_saved} samples to disk.")

                        # This assumes an increasing ordering of models by parameter count.
                        largest_model = model_categories[-1]
                        instances_to_propagate = updated_requests[largest_model]
                        result_score = result_scores[largest_model]
                        chosen_model_id = len(model_categories) - 1

                    else:
                        logger.info(f"Running request #{timestamp} - Estimating")
                        selected_doc = request_list[0].doc
                        sample = self.sample_generator.make_sample(
                            doc_id=doc_id,
                            input_data=selected_doc,
                            task=task,
                        )

                        preds, probs = self.mess_classifier.predict(texts=[sample["input_text"].item()])
                        energy = []
                        for model_category in self.measurements.keys():
                            energy.append(
                                sum(
                                    map(lambda x: sum([
                                        i for i in x.gpu_energy.values()
                                    ]) / len(self.measurements[model_category]), self.measurements[model_category])
                                )
                            )

                        energy = np.array(energy).reshape(-1, 1)
                        probs = probs.reshape(-1, 1)
                        cost_fn = self.algorithm_config["V"] * energy + self.Q * (
                                self.algorithm_config["alpha"] - probs
                        )

                        cost_fn = cost_fn.reshape(1, -1)
                        print(f"COST FN: {cost_fn}.")
                        chosen_model_id = np.argmin(cost_fn)
                        model_category_chosen = model_categories[chosen_model_id]

                        result = self.query_model(
                            request_list=request_list,
                            doc_id=doc_id,
                            selected_doc=selected_doc,
                            task=task,
                            model=self.vllm_models[model_category_chosen],
                            model_category=model_category_chosen,
                        )

                        instances_to_propagate = result["updated_requests"]
                        # TODO: The dict key at this point may have a different name for different benchmarks!
                        result_score = result[
                            self.sample_generator.benchmark_metrics_mapping[task.config.task.lower()]
                        ]

                    for instance in instances_to_propagate:
                        instances_by_doc_id[doc_id].append(instance)

                    self.Q = max(0.0, self.Q + self.algorithm_config["alpha"] - result_score)
                    self.score_list.append(result_score)
                    self.model_chosen_list.append(chosen_model_id)
                    self.num_exploration_steps.append(x_t)

                    print(f"Q LENGTH: {self.Q}.")

                    if wandb.run is not None:
                        wandb.log({
                            "mess/accuracy": sum(self.score_list) / (timestamp + 1),
                            "mess/model_chosen_ratio": sum(self.model_chosen_list) / (timestamp + 1),
                            "mess/exploration_rate": sum(self.num_exploration_steps) / (timestamp + 1),
                            "mess/q_length": self.Q,
                        }, step=timestamp)

                if self.classifier_config["generate_training_dataset"] and self.classifier_config["write_training_dataset_to_disk"]:
                    writer_stats = self.data_writer.finalize()
                    logger.info(f"Writing training dataset to disk done. Details: {writer_stats}")

        ### Postprocess outputs ###
        task.apply_filters()

        ### Collect values of metrics on all datapoints ###

        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)

        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps.keys():
            doc_iterator = task.doc_iterator(
                rank=0, limit=limit, world_size=1
            )

            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                results_input = []
                for req in requests:
                    if is_nested_list(req.filtered_resps[filter_key]):
                        results_input.append(req.filtered_resps[filter_key][0])
                    else:
                        results_input.append(req.filtered_resps[filter_key])

                metrics = task.process_results(doc, results_input)

                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in requests
                        ],
                        "filter": filter_key,
                        "metrics": list(metrics.keys()),
                        "doc_hash": hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                        "prompt_hash": hash_string(requests[0].arguments[0]),
                        "target_hash": hash_string(str(target)),
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)

        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)

        self.__evict_vllm_models()

        return None

    def query_model(self, request_list, model, model_category, doc_id, selected_doc, task) -> Dict[str, List]:
        assert type(request_list) is list, "Make sure to pass a list of requests to the query_model() function."
        self.energy_monitor.begin_window(f"pass_{model_category}")

        outputs = None
        results = {}
        try:
            outputs = getattr(
                model["vllm_eval_instance"],
                request_list[0].request_type
            )(
                request_list, disable_tqdm=True
            )

            for request, output in zip(request_list, outputs):
                request.resps.append((output[0], output[1]))

        except torch.OutOfMemoryError as e:
            logger.error(f"Encountered error processing document #{doc_id}: {e}")

        measurement = self.energy_monitor.end_window(f"pass_{model_category}")
        self.measurements[model_category].append(measurement)

        results.update({
            "updated_requests": request_list,
            "energy_consumption": sum([val for val in measurement.gpu_energy.values()]),
            "inference_time": measurement.time
        })

        if outputs:
            metrics = task.process_results(selected_doc, [(i[0], i[1]) for i in outputs])
            results.update(metrics)

        return results

    def __sample_from_bernoulli(self, c: float, timestamp: int):

        p_t = min(
            1.0, c / np.power(1 if timestamp == 0 else timestamp, (1/5))
        )

        x_t = binomial(n=1, p=p_t, size=1)

        self.wandb_run.log({
            "p_t": p_t,
            "x_t": x_t
        }, step=timestamp)

        return x_t.item()

    def __warm_up_inference_models(self):
        self.vllm_models = {}
        self.tokenizers = {}

        logger.info(f"Found {len(self.config['model_zoo'].keys())} models in zoo: {self.config['model_zoo'].keys()}")
        for model, data in self.config["model_zoo"].items():

            if data["category"] not in self.vllm_models.keys():
                self.vllm_models[data["category"]] = {}

            os.environ["CUDA_VISIBLE_DEVICES"] = str(data["gpu_indices"]).replace("[", "").replace("]", "")
            print(os.environ["CUDA_VISIBLE_DEVICES"])
            self.vllm_models[data["category"]] = {
                "model_name": model,
                "vllm_eval_instance": MessLMEvalVLLM(
                    model,
                    max_length=data["max_seq_len"],
                    gpu_indices=data["gpu_indices"],
                    trust_remote_code=True,
                    tensor_parallel_size=len(data["gpu_indices"]),
                    gpu_memory_utilization=data["gpu_memory_utilization"],
                    quantization=data["quantization"],
                    seed=self.config["seed"],
                    enforce_eager=self.lm_eval_config["enforce_eager"]
                )
            }

            logger.info(f"vLLM model {model} loaded on rank {data['gpu_indices']}. Tensor parallel size: {len(data['gpu_indices'])}")

        logger.info(f"All models loaded.")

    def __warmup_classifier_model(self):

        self.classifier_device = f"cuda:{self.classifier_config['gpu_index']}" if torch.cuda.is_available() else "cpu"

        self.mess_classifier = ContinualMultilabelBERTClassifier(
            model_name=self.classifier_config["model_id"],
            num_labels=len(self.vllm_models.keys()),
            learning_rate=self.classifier_config["learning_rate"],
            weight_decay=self.classifier_config["weight_decay"],
            batch_size=self.classifier_config["batch_size"],
            max_length=self.classifier_config["max_length"],
            warmup_ratio=self.classifier_config["warmup_ratio"],
            threshold=self.classifier_config["threshold"],
            dropout_rate=self.classifier_config["dropout_rate"],
            freeze_bert_layers=self.classifier_config["freeze_bert_layers"],
            device=self.classifier_device,
            config=self.classifier_config
        )

        logger.info(
            f"Classification model {self.config['classifier_model']['model_id']} loaded and ready to use. "
            f"Classifier model loaded onto device: {self.classifier_device}."
        )

        if self.classifier_config["use_pretrained_classifier"] is True:
            self.mess_classifier.load_model(path=f"{PROJECT_ROOT_PATH}/{self.classifier_config['checkpoint_path']}")
            logger.info(
                f"Using pre-trained classifier available under path "
                f"{PROJECT_ROOT_PATH}/{self.classifier_config['checkpoint_path']}"
            )

    def __evict_vllm_models(self):

        destroy_model_parallel()
        destroy_distributed_environment()

        models = [i for i in self.vllm_models.keys()]

        for model_category in models:
            del self.vllm_models[model_category]["vllm_eval_instance"].model.llm_engine.model_executor
            del self.vllm_models[model_category]

        gc.collect()
        torch.cuda.empty_cache()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        torch.cuda.synchronize()

    def __train_classifier_one_step(
        self,
        training_sample: pd.DataFrame,
        timestamp: int,
        label_col_identifier: str = "labels_"
    ):

        label_cols = [name for name in training_sample.columns if label_col_identifier in name]
        cols = ["input_text"] + label_cols
        data = training_sample[cols]

        dataset = BertPandasDataset(
            dataframe=data,
            text_col="input_text",
            y_cols=label_cols,
            tokenizer=self.mess_classifier.tokenizer,
            max_length=self.classifier_config["max_length"]
        )

        self.energy_monitor.begin_window(f"classifier_training_step")
        self.mess_classifier.incremental_fit(
            new_train_dataset=dataset,
            # new_val_dataset=dataset,
            epochs=self.classifier_config["epochs"],
            memory_strategy=self.classifier_config["memory_strategy"],
            learning_rate=self.classifier_config["learning_rate"],
            reset_optimizer=self.classifier_config["reset_optimizer"],
            regularization_lambda=self.classifier_config["regularization_lambda"],
            timestamp=timestamp
        )
        measurement = self.energy_monitor.end_window(f"classifier_training_step")
        self.wandb_run.log({
            "train/step_energy": sum([v for v in measurement.gpu_energy.values()])
        }, step=timestamp)

        training_sample["classifier_energy_consumption"] = sum([val for val in  measurement.gpu_energy.values()])

        return training_sample

    def __adjust_config(self, task_dict, gen_kwargs, predict_only, num_fewshot, fewshot_random_seed):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: self.__adjust_config(
                        task_obj,
                        gen_kwargs=gen_kwargs,
                        predict_only=predict_only,
                        num_fewshot=num_fewshot,
                        fewshot_random_seed=fewshot_random_seed
                    )},
                }

            else:
                if task_obj.get_config("output_type") == "generate_until":
                    if gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=gen_kwargs, update=True
                        )

                if predict_only:
                    logger.info(
                        f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                    )
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        logger.info(
                            f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                        )
                    else:
                        logger.warning(
                            f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                        )
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (
                            default_num_fewshot := task_obj.get_config("num_fewshot")
                    ) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    def __get_training_sample(self, request_list: List[Instance], doc_id: int, task: Task) -> tuple[dict[Any, Any], dict[Any, Any], Any]:
        outputs = {i: dict() for i in self.vllm_models.keys()}
        selected_doc = request_list[0].doc

        for model_category, model in self.vllm_models.items():
            outputs[model_category] = self.query_model(
                request_list=request_list,
                model=model,
                model_category=model_category,
                doc_id=doc_id,
                task=task,
                selected_doc=selected_doc,
            )

        sample = self.sample_generator.make_sample(
            doc_id=doc_id,
            input_data=selected_doc,
            model_response_data=outputs,
            task=task,
            stage="train"
        )

        updated_inference_requests = {model_category: data["updated_requests"] for model_category, data in outputs.items()}
        result_scores = {
            model_category: data[
                self.sample_generator.benchmark_metrics_mapping[task.config.task.lower()]
            ] for model_category, data in outputs.items()
        }
        return updated_inference_requests, result_scores, sample

    @staticmethod
    def validate_tasks(lm, eval_tasks, confirm_run_unsafe_code: bool = True):
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

    def make_wandb_run_name(self, task):
        name = f"{task.task_name}"

        if self.classifier_config["generate_training_dataset"]:
            return name + "training-dataset-generation"

        name += f"a={self.algorithm_config['alpha']}_"
        name += f"c={self.algorithm_config['c']}_"
        name += f"V={self.algorithm_config['V']}"
        name += f"classif-pretrained={self.classifier_config['use_pretrained_classifier']}"

        return name

    def shutdown(self):
        self.__evict_vllm_models()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MESS+ Algorithm Executor with LM-Eval integration")
    parser.add_argument(
        "-c",
        "--config",
        type=str
    )

    parser.add_argument(
        "-e",
        "--wandb-entity",
        type=str
    )

    parser.add_argument(
        "-p",
        "--project-name",
        type=str
    )

    args = parser.parse_args()

    selector = MessPlusAutomaticModelSelector(
        config_file_path=args.config,
        project_name=args.project_name,
        wandb_entity=args.wandb_entity
    )

    try:
        selector.launch()
    except KeyboardInterrupt or AttributeError:
        selector.shutdown()
