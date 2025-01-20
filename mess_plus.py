import contextlib
import gc
import logging
import os
import time

import pandas as pd
import torch
import numpy as np
import yaml

import torch.nn as nn

from numpy.random import binomial
from vllm import LLM, SamplingParams

from datasets import Dataset

from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

from sklearn.metrics import accuracy_score, f1_score

from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

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
from lm_eval.tasks import (
    Task,
    TaskManager,
    get_task_dict,
)
from utils.mess_lm_eval_harness.vllm import MessLMEvalVLLM


from utils.modelling_messplus_classifier import make_mlp
from classifier.utils.lit_trainer import MESSPlusTrainer

from zeus.monitor import ZeusMonitor

from collections import defaultdict
from typing import List, Tuple, Optional, Type, Callable

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


class MessPlusAutomaticModelSelector(object):

    def __init__(self, config_file_path: str):
        self.config = yaml.safe_load(open(config_file_path, "r"))
        self.lm_eval_config = self.config["lm_eval"]
        self.dataset = None
        self.input_column_name = None
        self.expected_response_column_name = None

        self.__warm_up_inference_models()

        # Classifier model
        self.__warmup_classifier_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Loggers
        self.processing_times = dict()
        self.model_energy_consumption = dict()

        # When using Zeus, you must disable RAPL CPU monitoring as this will cause the program to fail.
        # Change "True" to "False" in file venv/lib/python3.12/site-packages/zeus/device/cpu/rapl.py (l. 137)
        self.energy_monitor = ZeusMonitor(gpu_indices=[i for i in range(NUM_GPUS)])

        # LM Eval Config
        task_manager = TaskManager(verbosity="INFO")

        self.task_dict = get_task_dict(self.lm_eval_config["benchmarks"], task_manager)
        self.task_dict = self.__adjust_config(
            self.task_dict,
            gen_kwargs=self.lm_eval_config["gen_kwargs"] if "gen_kwargs" in self.config.keys() else None,
            predict_only=False,
            num_fewshot=0,
            fewshot_random_seed=self.config["seed"]
        )

        self.eval_tasks = get_task_list(self.task_dict)

    def run_benchmark(
        self,
        dataset: pd.DataFrame,
        text_col: str,
        label_col: str,
        limit_num_samples: int = None,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Optional[Callable] = None,
        tokenizer_name: str = "",
        write_out: bool = False,
        log_samples: bool = False
    ) -> None:
        """
        This function expects a pandas dataframe where there is a text input and a corresponding label.
        :param dataset: pandas DataFrame containing the dataset.
        :param text_col: Name of the text column
        :param label_col: Name of the label column
        :param limit_num_samples: Number of samples to use for MESS+
        :param
        :return: None
        """

        if apply_chat_template:
            logger.warning(
                "Chat template formatting change affects loglikelihood and multiple-choice tasks. See docs/chat-template-readme.md for details."
            )

        # tracks all Instances/requests a model must generate output on.
        requests = defaultdict(list)
        # stores the amount to pad out reqs per req. type so that
        # number of fwd passes per distributed rank is equal
        padding_requests = defaultdict(int)

        # get lists of group hierarchy and each type of request
        eval_tasks = get_task_list(self.task_dict)
        if not log_samples:
            if not all(
                    "bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys()
                    for task_output in eval_tasks
            ):
                raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

        limit_arg = limit_num_samples
        limits = []
        for task_output in self.eval_tasks:
            task: Task = task_output.task

            smallest_model_category, smallest_model_instance = next(iter(self.vllm_models.items()))
            smallest_model_instance = smallest_model_instance["vllm_eval_instance"]

            limit = get_sample_size(task, limit_arg)
            limits.append(limit)
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
            for instance in task.instances:
                reqtype = instance.request_type
                requests[reqtype].append(instance)

        ### Run LM on inputs, get all outputs ###
        # execute each type of request
        for reqtype, reqs in requests.items():
            logger.info(f"Running {reqtype} requests")
            # create `K` copies of each request `req` based off `K = req.repeats`
            cloned_reqs = []
            for req in reqs:
                cloned_reqs.extend([req] * req.repeats)

            model_resps_mapping = {i: [] for i in self.vllm_models.keys()}
            for model_category, model in self.vllm_models.items():
                model_resps_mapping[model_category] = getattr(model["vllm_instance"], reqtype)(cloned_reqs)

            # for model_category, resps in model_resps_mapping.items():
            #     # put responses from model into a list of length K for each request.
            #     for x, req in zip(resps, cloned_reqs):
            #         req.resps.append(x)

            print(model_resps_mapping)

        # df = dataset[[text_col, label_col]]
        # df.reset_index(inplace=True)
        # # We need the index to start at 1.
        # df.index += 1
        #
        # Q = 0.0
        # E = [1, 8, 70]  # Dummy energy consumption for Llama 1b, 8b, 70b
        # alpha = self.config["algorithm"]["alpha"]
        #
        # for idx, row in df.iterrows():
        #
        #     print(row[text_col])
        #     print(row[label_col])
        #
        #     chosen_response, m_acc = self.run_request(
        #         req=row[text_col],
        #         ground_truth=row[label_col],
        #         timestamp=idx,
        #         c=self.config["algorithm"]["c"],
        #         V=self.config["algorithm"]["V"],
        #         alpha=alpha,
        #         E=E, Q=Q
        #     )
        #
        #     Q = max(0.0, Q + alpha - m_acc)
        #
        #     if idx % 10 == 0:
        #         logger.info(f"Processed indices {idx - 10} to {idx}.")
        #
        # logger.info("Dataset processing complete.")

    def run_request(self, req: str, ground_truth: str, timestamp: 
                    int, c: float, V: float, alpha: float, E: list, Q: float):
        x_t = self.__sample_from_bernoulli(c=c, timestamp=timestamp)

        if x_t == 1:
            # Let's explore
            logger.info(f"Exploring during step {timestamp}.")

            responses = []
            preference_scores = []

            for category in self.vllm_models.keys():

                embedded_req = self.__add_system_prompt_and_chat_template(req, category)

                self.energy_monitor.begin_window("pass")
                response = self.vllm_models[category]["vllm_instance"].generate(embedded_req)
                measurement = self.energy_monitor.end_window("pass")

                response = response[0].outputs[0].text.strip()

                # It is important that this list is ordered.
                responses.append(response)
                preference_scores.append(1 if response == ground_truth else 0)

                logger.info(f"The request consumed a total energy of {measurement.gpu_energy} joules.")
                logger.info(f"The request took {measurement.time} seconds to complete.")

            tokenized_input = self.classifier_tokenizer(
                req,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            labels = self.__prepare_classifier_training_labels(preference_scores)
            print(labels)

            dataset = Dataset.from_dict({
                "input_ids": tokenized_input["input_ids"],
                "attention_mask": tokenized_input["attention_mask"],
                "labels": labels
            })

            # Prepare the classifier training environment
            self.__evict_vllm_models()

            # Train the classifier
            self.train_classifier(data=dataset)

            # Need to re-initiate inference models
            self.__warm_up_inference_models()

            # We return the output of the largest (and most powerful) model.
            chosen_response = responses[-1]

        else:
            # Let's estimate
            preferences = self.estimate_user_preferences(req=req)

            if isinstance(preferences, torch.Tensor):
                preferences = preferences.detach().cpu().tolist()

            costs = []
            model_keys = list(self.vllm_models.keys()) 

            # Compute cost for each model: stochastic optimization
            for i, pref in enumerate(preferences):
                m_cost = V * E[i] + Q * (alpha - pref)
                costs.append(m_cost)

            min_cost_idx = int(np.argmin(costs))
            chosen_category = model_keys[min_cost_idx]

            logger.info(f"Chosen model is '{chosen_category}' with cost = {costs[min_cost_idx]:.4f}")
            
            # Get response from chosen model
            embedded_req = self.__add_system_prompt_and_chat_template(req, chosen_category)
            self.energy_monitor.begin_window("pass")
            response_obj = self.vllm_models[chosen_category]["vllm_instance"].generate(embedded_req)
            measurement = self.energy_monitor.end_window("pass")

            chosen_response = response_obj[0].outputs[0].text.strip()

            logger.info(f"Model '{chosen_category}' response: {chosen_response}")
            logger.info(f"Energy usage (Joules): {measurement.gpu_energy}, Time (sec): {measurement.time}")
   
        m_acc = 1.0 if chosen_response == ground_truth else 0.0
        return chosen_response, m_acc

    @staticmethod
    def __sample_from_bernoulli(c: float, timestamp: int):

        p_t = min(
            1.0, c / np.cbrt(timestamp)
        )

        x_t = binomial(n=1, p=p_t, size=1)

        return x_t.item()

    def __warm_up_inference_models(self):
        self.vllm_models = {}
        self.tokenizers = {}

        # This is a safeguard so that we do not over-commit GPUs and cause unexpected OOMs.
        max_memory_utilization_per_model = min(1.0, torch.cuda.device_count() / len(self.config["model_zoo"].keys()))

        for model, data in self.config["model_zoo"].items():

            if data["category"] not in self.vllm_models.keys():
                self.vllm_models[data["category"]] = {}

            os.environ["CUDA_VISIBLE_DEVICES"] = str(data["gpu_indices"]).replace("[", "").replace("]", "")
            self.vllm_models[data["category"]] = {
                "model_name": model,
                "vllm_eval_instance": MessLMEvalVLLM(
                    model,
                    max_seq_len=data["max_seq_len"],
                    gpu_indices=data["gpu_indices"],
                    trust_remote_code=True,
                    tensor_parallel_size=len(data["gpu_indices"]),
                    max_memory_utilization=max_memory_utilization_per_model
                ),
                "tokenizer": AutoTokenizer.from_pretrained(model)
            }

            logger.info(f"vLLM model {model} loaded on rank {data['gpu_indices']}. Tensor parallel size: {len(data['gpu_indices'])}")

        logger.info(f"All models loaded.")

    def __warmup_classifier_model(self):
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
            self.config["classifier_model"]["model_id"],
            num_labels=self.config["classifier_model"]["num_labels"]
        )

        self.__make_mlp_classifier()

        self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.config["classifier_model"]["model_id"])
        self.classifier_tokenizer.model_max_length = self.config["classifier_model"]["max_seq_len"]

        logger.info(f"Classification model {self.config['classifier_model']['model_id']} loaded and ready to use.")

    def __evict_vllm_models(self):

        destroy_model_parallel()
        destroy_distributed_environment()

        models = [i for i in self.vllm_models.keys()]

        for model_category in models:
            del self.vllm_models[model_category]["vllm_instance"].llm_engine.model_executor
            del self.vllm_models[model_category]

        gc.collect()
        torch.cuda.empty_cache()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        torch.cuda.synchronize()

    def __add_system_prompt_and_chat_template(self, req, model_category: str):
        message = [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": req}
        ]

        return self.vllm_models[model_category]["tokenizer"].apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

    @staticmethod
    def __prepare_classifier_training_labels(preference_scores: list):
        """
        Returns the index of the smallest model that was able to respond correctly to the question.
        :param preference_scores:
        :return: Tensor with the smallest model index that was able to solve the question.
        """
        if 1 in preference_scores:
            up_to_first_one = preference_scores[:preference_scores.index(1) + 1]
            after_first_one = [0 if x == 1 else x for x in preference_scores[preference_scores.index(1) + 1:]]

            return up_to_first_one + after_first_one
        else:
            return preference_scores.copy()

    # @staticmethod
    # def __append_label_for_human_annotation(preference_scores: list, val: int = 1):
    #     # We do not consider human annotation an option at this point.
    #     preference_scores += [val]
    #     return preference_scores

    def __make_mlp_classifier(self):

        # We freeze the backbone model parameters
        for param in self.classifier_model.model.parameters():
            param.requires_grad = False

        self.classifier_model.classifier = make_mlp(
            base_model=self.classifier_model,
            config=self.config
        )

        trainable_parameters = filter(lambda p: p.requires_grad, self.classifier_model.parameters())
        param_count = sum([np.prod(p.size()) for p in trainable_parameters])

        logger.info(f"Using a classification MLP with {param_count} trainable parameters.")

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

    def validate_tasks(self, lm, eval_tasks, confirm_run_unsafe_code: bool = True):
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

    def train_classifier(self, data):
        self.classifier_model = self.classifier_model.to(self.device)
        self.classifier_model.config.problem_type = "single_label_classification"

        training_args = TrainingArguments(
            output_dir="messplus_modernbert",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=5e-5,
            num_train_epochs=5,
            bf16=True,
            optim="sgd",
            logging_strategy="steps",
            logging_steps=1,
            eval_strategy="steps",
            report_to="wandb",
        )

        trainer = WeightedLossTrainer(
            model=self.classifier_model,
            args=training_args,
            train_dataset=data,
            eval_dataset=data,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        self.classifier_model = self.classifier_model.to("cpu")

    def estimate_user_preferences(self, req):
        tokenized_inputs = self.classifier_tokenizer(
            [req],
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.classifier_model(**tokenized_inputs)

        preds = outputs.logits
        probs = torch.softmax(preds, dim=-1)

        estimated_user_preferences = []
        for idx in range(len(self.config["model_zoo"].keys())):
            # For now, this only works for single requests. When using batched processing, we need to introduce another
            # dimension here.
            if probs.shape != torch.Size([1, probs.shape[-1]]):
                logger.warning(f"It seems you are trying to use batched inputs. Currently the system only supports "
                               f"single requests. Change the way probs are summed up.")

            estimated_user_preferences.append(
                torch.sum(probs[:, 0:(idx + 1)])
            )

        return estimated_user_preferences

    @staticmethod
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        print(predictions, labels)
        predictions = np.argmax(predictions, axis=1)

        f1 = f1_score(labels, predictions, average="weighted")
        accuracy = accuracy_score(labels, predictions)

        metrics = {
            "f1": float(f1) if f1 == 1 else f1,
            "accuracy": float(accuracy),
        }

        logger.info(f"Evaluated model with metrics: {metrics}")

        return metrics

if __name__ == "__main__":

    dummy_dataset = [
        {"text": "You are given two options A and B. Option A is correct.", "label": "A"},
        {"text": "You are given two options A and B. Option B is correct.", "label": "B"},
        {"text": "You are given two options A and B. Option A is correct.", "label": "A"},
        {"text": "You are given two options A and B. Option B is correct.", "label": "B"},
        {"text": "You are given two options A and B. Option A is correct.", "label": "A"},
        {"text": "You are given two options A and B. Option B is correct.", "label": "B"},
        {"text": "You are given two options A and B. Option A is correct.", "label": "A"},
        {"text": "You are given two options A and B. Option B is correct.", "label": "B"},
        {"text": "You are given two options A and B. Option A is correct.", "label": "A"},
        {"text": "You are given two options A and B. Option B is correct.", "label": "B"},
        {"text": "You are given two options A and B. Option A is correct.", "label": "A"},
        {"text": "You are given two options A and B. Option B is correct.", "label": "B"},
        {"text": "You are given two options A and B. Option A is correct.", "label": "A"},
        {"text": "You are given two options A and B. Option B is correct.", "label": "B"},
        {"text": "You are given two options A and B. Option A is correct.", "label": "A"},
        {"text": "You are given two options A and B. Option B is correct.", "label": "B"},
        {"text": "You are given two options A and B. Option A is correct.", "label": "A"},
        {"text": "You are given two options A and B. Option B is correct.", "label": "B"},
        {"text": "You are given two options A and B. Option A is correct.", "label": "A"},
        {"text": "You are given two options A and B. Option B is correct.", "label": "B"},
    ]

    df = pd.DataFrame(dummy_dataset)

    config_file_path = "config/messplus/boolq_baseline.yaml"

    selector = MessPlusAutomaticModelSelector(
        config_file_path=config_file_path
    )

    selector.run_dataset(
        dataset=df,
        text_col="text",
        label_col="label"
    )
