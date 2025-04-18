import argparse
import contextlib
import gc
import json
import logging
import os

import pandas as pd
import transformers
import torch
import numpy as np
import wandb
import yaml

from numpy.random import binomial
from torch.utils.data import DataLoader
from torch.nn import functional as F
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

from utils.data_capturing import StreamingDataProcessor, DataExtractor
from utils.mess_lm_eval_harness.vllm_v2 import MessLMEvalVLLM
from classifier.model import ContinualMultilabelBERTClassifier
from classifier.dataset import BertPandasDataset

from zeus.monitor import ZeusMonitor

from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'mess_plus.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
NUM_GPUS = torch.cuda.device_count()

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision('high')


class MessPlusAutomaticModelSelector(object):

    def __init__(self, config_file_path: str, project_name):
        self.config = yaml.safe_load(open(config_file_path, "r"))
        self.lm_eval_config = self.config["lm_eval"]
        self.algorithm_config = self.config["algorithm"]
        self.wandb_project_name = project_name

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
        self.measurements = {d["category"]: [] for i, d in self.config["model_zoo"].items()}
        self.ground_truths = {d["category"]: [] for i, d in self.config["model_zoo"].items()}
        self.predictions = {d["category"]: [] for i, d in self.config["model_zoo"].items()}
        self.scores = {d["category"]: [] for i, d in self.config["model_zoo"].items()}
        self.energy_monitor = ZeusMonitor(gpu_indices=[i for i in range(NUM_GPUS)], approx_instant_energy=True)
        self.classifier_running_train_loss = 0.0
        self.classifier_train_steps = 0
        self.classifier_running_val_loss = 0.0
        self.classifier_val_steps = 0
        self.algorithm_correct_choices = 0

        # Algorithm config
        # Q is a virtual queue, i.e., we only keep the sum of all violations, no history.
        self.Q = 0.0

        # LM Eval Config
        task_manager = TaskManager(verbosity="INFO")

        self.limit_num_samples = self.lm_eval_config["limit_num_samples"]
        self.limits = []

        self.task_dict = get_task_dict(self.lm_eval_config["benchmarks"], task_manager)
        self.task_dict = self.__adjust_config(
            self.task_dict,
            gen_kwargs=self.lm_eval_config["gen_kwargs"] if "gen_kwargs" in self.config.keys() else None,
            predict_only=False,
            num_fewshot=0,
            fewshot_random_seed=self.config["seed"]
        )

        self.eval_tasks = get_task_list(self.task_dict)

        # Config to capture the inference outputs for classifier validation
        self.infer_data_processor = StreamingDataProcessor(
            save_path=f"data/inference_outputs/",
            file_prefix="inference_data_",
            save_frequency=100
        )

        self.reference_data_reader = DataExtractor()


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
            # This is where plugin the MESS+ custom routine.
            self.run_benchmark(task_output, limit_arg)

        print(self.eval_tasks)
        results, samples, configs, versions, num_fewshot, higher_is_better, = consolidate_results(self.eval_tasks)

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
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        ### Run LM on inputs, get all outputs ###
        # execute each type of request        
        for reqtype, reqs in requests.items():
            logger.info(f"Processing {reqtype} requests for {task.task_name} dataset with a total of {len(requests[reqtype])} samples.")
            # create `K` copies of each request `req` based off `K = req.repeats`
            cloned_reqs = []
            for req in reqs:
                cloned_reqs.extend([req] * req.repeats)

            logger.info(f"Dataset repeats factor: {len(cloned_reqs) / len(requests[reqtype])}")
            with wandb.init(
                project=self.wandb_project_name,
                name=f"{self.config['lm_eval']['benchmarks'][0]}",
                config=self.config
            ) as run:
                self.wandb_run = run
                self.wandb_run.summary["V"] = self.algorithm_config["V"]
                self.wandb_run.summary["alpha"] = self.algorithm_config["alpha"]
                self.wandb_run.summary["c"] = self.algorithm_config["c"]

                for idx, request in enumerate(cloned_reqs):
                    outputs = self.run_request(request, timestamp=idx)
                    reference_data = self.reference_data_reader.get_data(
                        benchmark_name=f"{self.config['lm_eval']['benchmarks'][0]}",
                        request=request
                    )
                    self.infer_data_processor.process_row(
                        outputs,
                        input_text=reference_data["input_data"],
                        ground_truth=reference_data["ground_truth"],
                        doc_id=reference_data["doc_id"],
                        benchmark_name=f"{self.config['lm_eval']['benchmarks'][0]}",
                        write_to_disk=self.lm_eval_config["write_to_disk"]
                    )

                # We initialize one classifier model for every benchmark
                # self.__warmup_classifier_model()
                # for idx, request in enumerate(cloned_reqs):
                #     _, _, output = self.run_request(request, timestamp=idx)
                #     for out in output:
                #         request.resps.append(
                #             (out[0], out[1])
                #         )

        # ### Postprocess outputs ###
        # task.apply_filters()
        #
        # ### Collect values of metrics on all datapoints ###
        # # # unpack results and sort back in order and return control to Task
        # # Pre-process task.instances to group by doc_id
        # instances_by_doc_id = defaultdict(list)
        # for instance in task.instances:
        #     instances_by_doc_id[instance.doc_id].append(instance)
        # # Sort instances within each group
        # for instances in instances_by_doc_id.values():
        #     instances.sort(key=lambda x: x.idx)
        # # iterate over different filters used
        # for filter_key in task.instances[0].filtered_resps.keys():
        #     doc_iterator = task.doc_iterator(
        #         rank=0, limit=limit, world_size=1
        #     )
        #
        #     for doc_id, doc in doc_iterator:
        #         requests = instances_by_doc_id[doc_id]
        #         results_input = []
        #         for req in requests:
        #             if is_nested_list(req.filtered_resps[filter_key]):
        #                 results_input.append(req.filtered_resps[filter_key][0])
        #             else:
        #                 results_input.append(req.filtered_resps[filter_key])
        #
        #         metrics = task.process_results(doc, results_input)
        #         if log_samples:
        #             target = task.doc_to_target(doc)
        #             example = {
        #                 "doc_id": doc_id,
        #                 "doc": doc,
        #                 "target": target,
        #                 "arguments": [req.args for req in requests],
        #                 "resps": [req.resps for req in requests],
        #                 "filtered_resps": [
        #                     req.filtered_resps[filter_key] for req in requests
        #                 ],
        #                 "filter": filter_key,
        #                 "metrics": list(metrics.keys()),
        #                 "doc_hash": hash_string(
        #                     json.dumps(
        #                         requests[0].doc,
        #                         indent=2,
        #                         default=handle_non_serializable,
        #                         ensure_ascii=False,
        #                     )
        #                 ),
        #                 "prompt_hash": hash_string(requests[0].arguments[0]),
        #                 "target_hash": hash_string(str(target)),
        #             }
        #             example.update(metrics)
        #             task_output.logged_samples.append(example)
        #         for metric, value in metrics.items():
        #             task_output.sample_metrics[(metric, filter_key)].append(value)
        #
        # ### Aggregate results over all datapoints ###
        # # aggregate results ; run bootstrap CIs
        # task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)

        self.infer_data_processor.finalize(write_to_disk=self.lm_eval_config["write_to_disk"])
        self.__evict_vllm_models()

        return None

    def query_model(self, model, request, model_category) -> Tuple[List[float], List[str], List[bool]]:
        self.energy_monitor.begin_window(f"pass_{model_category}")

        try:
            output = getattr(
                model["vllm_eval_instance"],
                request.request_type
            )(
                [request] if type(request) is not list else request, disable_tqdm=True
            )
        except torch.OutOfMemoryError as e:
            output = None
            logger.error(f"Encountered error processing document #{request.doc_id}: {e}")

        measurement = self.energy_monitor.end_window(f"pass_{model_category}")
        self.measurements[model_category].append(measurement)

        nll_scores = []
        responses = []
        correct = []
        if output is not None:
            ground_truth = request.arguments[1].strip().lower()
            for out in output:
                nll_scores.append(out[0])
                responses.append(out[2])
                correct.append(bool(out[2] == ground_truth))

                self.ground_truths[model_category].append(ground_truth)
                self.predictions[model_category].append(out[2])
                self.scores[model_category].append(out[0])

        return nll_scores, responses, correct

    def run_request(self, request: Instance, timestamp: int = 0):
        model_categories = [i for i in self.vllm_models.keys()]
        x_t = self.__sample_from_bernoulli(c=self.algorithm_config["c"], timestamp=timestamp)

        logger.info(f"Running request #{timestamp}...")

        outputs = self.__get_inference_model_labels(request=request)

        print(outputs)

        # We only keep numerical values (log_likelihood score and whether the generated response was correct)
        # training_label = []
        #
        # for model, data in outputs.items():
        #     training_label.extend(data[0])
        #     training_label.extend([float(i) for i in data[2]])

        return outputs

        # if x_t == True:
        #     logger.info(f"Exploring during step {timestamp}.")
        #
        #     outputs = self.__get_inference_model_labels(request=request)
        #
        #     # We only keep numerical values (log_likelihood score and whether the generated response was correct)
        #     training_label = []
        #
        #     for model, data in outputs.items():
        #         training_label.extend(data[0])
        #         training_label.extend([float(i) for i in data[2]])
        #
        #     print(f"Captured output: {training_label}.")
        #
        #     # label = torch.tensor(label)
        #     # model_category = [i for i in self.vllm_models.keys()][-1]
        #     # final_response = output[0][-1]
        #
        #     # Train the classifier
        #     # cp_measurement = self.__train_classifier_one_step(request=request, label=label.tolist(), timestep=timestamp)
        #     # gpu_keys = [i for i in cp_measurement.gpu_energy.keys()]
        #
        #     # The last GPU is always used to train an inference run for the classifier model.
        #     # logs = {"energy/train_classifier": cp_measurement.gpu_energy[gpu_keys[-1]]}
        #     # logs.update({
        #     #     f"preds/infer_{cat}": label[idx] for idx, cat in enumerate(model_categories)
        #     # })
        #     #
        #     # for idx, cat in enumerate(self.vllm_models.keys()):
        #     #     logs.update({f"energy/infer_{cat}": sum([v for v in self.measurements[cat][-1].gpu_energy.values()])})
        #     #
        #     # logs.update({f"time/infer_{cat}": self.measurements[cat][-1].time for idx, cat in enumerate(model_categories)})
        #     # logs.update({
        #     #     f"algorithm/infer_response_{model_category}": final_response
        #     # })
        #     #
        #     # self.wandb_run.log(logs, step=timestamp)
        #
        # else:
        #     logger.info(f"Estimating during step {timestamp}.")
        #     logs = {}
        #
        #     preference_estimation, inference_model_label = self.estimate_user_preferences(
        #         request=request,
        #         timestamp=timestamp
        #     )
        #
        #     logs.update({
        #         f"algorithm/satisfaction_likelihood_{cat}": preference_estimation[idx] for idx, cat in enumerate(
        #             self.vllm_models.keys()
        #         )
        #     })
        #
        #     preference_estimation = preference_estimation.reshape(-1, 1)
        #     energy = []
        #     for model_category in self.measurements.keys():
        #         energy.append(
        #             sum(map(lambda x: sum([i for i in x.gpu_energy.values()]) / len(self.measurements[model_category]), self.measurements[model_category]))
        #         )
        #
        #     logs.update({f"algorithm/energy_estimate_{cat}": energy[idx] for idx, cat in enumerate(self.vllm_models.keys())})
        #     energy = torch.tensor(energy).reshape(-1, 1)
        #
        #     # Energy and Preference Estimation are vectors.
        #     cost_fn = self.algorithm_config["V"] * energy + self.Q * (self.algorithm_config["alpha"] - preference_estimation)
        #     logs.update({f"algorithm/cost_fn_{cat}": cost_fn[idx].item() for idx, cat in enumerate(self.vllm_models.keys())})
        #
        #     choice_idx = torch.argmax(cost_fn)
        #     choice_is_correct = 1 if torch.tensor(inference_model_label).reshape(-1, 1)[choice_idx].item() == 1 else 0
        #     self.algorithm_correct_choices += choice_is_correct
        #     logs.update({
        #         f"algorithm/chosen_model": choice_idx,
        #         f"algorithm/choice_correct": choice_is_correct,
        #         f"algorithm/choice_accuracy": float(self.algorithm_correct_choices / timestamp),
        #         f"algorithm/q_size": self.Q
        #     })
        #     model_category = model_categories[choice_idx]
        #
        #     # Query the model of choice
        #     # This is what the output looks like: [(-1.7801642417907715, False)] (neg log-likelihood, Response)
        #     output = self.query_model(
        #         self.vllm_models[model_category],
        #         request,
        #         model_category
        #     )
        #
        #     final_response = int(output[0][1])
        #     logs.update({f"algorithm/infer_response_{model_category}": final_response})
        #     logs.update({f"algorithm/infer_energy_{model_category}": sum([v for v in self.measurements[model_category][-1].gpu_energy.values()])})
        #
        #     self.wandb_run.log(logs, step=timestamp)
        #
        # # Queue update based on satisfaction rate.
        # # self.Q = max(0.0, self.Q + self.algorithm_config["alpha"] - final_response)
        #
        # return True, True, True  # final_response, model_category, output

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
            device=self.classifier_device
        )

        logger.info(
            f"Classification model {self.config['classifier_model']['model_id']} loaded and ready to use. "
            f"Classifier model loaded onto device: {self.classifier_device}."
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
        request: Instance,
        labels: dict[str, int],
        timestep: int,
        benchmark_name: str
    ):

        relevant_data = {
            "doc_id": request.doc_id,
            "input_data": request.doc['question']
        }

        # Add labels to data point.
        relevant_data.update(labels)

        pd_relevant_data = pd.DataFrame(relevant_data)
        dataset = BertPandasDataset(
            dataframe=pd_relevant_data.loc[timestep],
            text_col="input_text",
            y_cols=[i for i in labels.keys()],
            tokenizer=self.mess_classifier.tokenizer,
            max_length=self.classifier_config["max_length"]
        )

        self.energy_monitor.begin_window(f"classifier_training_step")
        self.mess_classifier.incremental_fit(
            new_train_dataset=dataset,
            # We are validating on the online sample at time t to capture the outcome quality.
            new_val_dataset=dataset,
            timestamp=timestep
        )
        measurement = self.energy_monitor.end_window(f"classifier_training_step")
        self.wandb_run.log({
            "train/step_energy": sum([v for v in measurement.gpu_energy.values()])
        }, step=timestep)

        return measurement

    def __adjust_config(self, task_dict, gen_kwargs, predict_only, num_fewshot, fewshot_random_seed):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: self.__adjust_config(task_obj)},
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

    def __get_inference_model_labels(self, request) -> dict[Any, tuple[Any]]:
        outputs = {i: tuple() for i in self.vllm_models.keys()}
        for model_category, model in self.vllm_models.items():
            outputs[model_category] = self.query_model(
                model,
                request,
                model_category
            )

        logger.debug(f"Model outputs: {outputs}.")

        return outputs

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

    def estimate_user_preferences(self, request: Instance, timestamp: int):
        # We need the true labels to estimate the predictor quality.
        label, _ = self.__get_inference_model_labels(request=request)

        dataset = ClassificationDataset(
            x=[request.doc["passage"] if "passage" in request.doc else request.doc["text"]],
            y=[label],
            tokenizer=self.classifier_tokenizer
        )

        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        self.energy_monitor.begin_window(f"classifier_prediction_step")
        stepper = 0
        step_loss = 0.0

        preference = None
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(self.classifier_device)
            attention_mask = batch['attention_mask'].to(self.classifier_device)
            labels = batch['label'].to(self.classifier_device)

            with torch.no_grad():
                outputs = self.mess_classifier(input_ids, attention_mask)
                loss = self.classifier_criterion(outputs, labels.float())

            step_loss += loss.item()
            stepper += 1
            self.classifier_running_val_loss += loss.item()
            self.classifier_val_steps += 1

            # We normalize the tensor to 1.0 such that we get the preference scores.
            norm_outputs = F.normalize(outputs.squeeze(), p=1.0, dim=0)
            preference = torch.cumsum(norm_outputs, dim=0).squeeze().cpu()

        measurement = self.energy_monitor.end_window(f"classifier_prediction_step")

        self.wandb_run.log({
            "classifier/pred_step_loss": float(step_loss / stepper),
            "classifier/pred_running_loss": float(self.classifier_running_train_loss / self.classifier_train_steps),
            "classifier/pred_step_energy": sum([v for v in measurement.gpu_energy.values()])
        }, step=timestamp)

        return preference, label

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
        "-p",
        "--project-name",
        type=str
    )

    args = parser.parse_args()

    selector = MessPlusAutomaticModelSelector(
        config_file_path=args.config,
        project_name=args.project_name
    )

    try:
        selector.launch()
    except KeyboardInterrupt or AttributeError:
        selector.shutdown()
