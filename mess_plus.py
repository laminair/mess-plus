import contextlib
import gc
import logging
import os

import pandas as pd
import transformers
import torch
import numpy as np
import pytorch_lightning as pl
import yaml

from numpy.random import binomial
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from sklearn.metrics import accuracy_score, f1_score

from lm_eval.evaluator_utils import get_sample_size, get_task_list,	print_writeout
from lm_eval.tasks import Task, TaskManager, get_task_dict
from lm_eval.api.task import Instance

from utils.mess_lm_eval_harness.vllm_v2 import MessLMEvalVLLM
from classifier.utils import MESSRouter, MESSOnlineLightningDataloader, MESSPlusTrainer, tokenize_request

from zeus.monitor import ZeusMonitor

from collections import defaultdict
from typing import List, Optional

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

os.environ[" TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision('high')


class MessPlusAutomaticModelSelector(object):

    def __init__(self, config_file_path: str):
        self.config = yaml.safe_load(open(config_file_path, "r"))
        self.lm_eval_config = self.config["lm_eval"]
        self.algorithm_config = self.config["algorithm"]
        self.dataset = None
        self.input_column_name = None
        self.expected_response_column_name = None

        self.__warm_up_inference_models()

        # Classifier model
        self.classifier_config = self.config["classifier_model"]
        self.__warmup_classifier_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.label_history = pd.DataFrame()

        # Loggers
        # When using Zeus, you must disable RAPL CPU monitoring as this will cause the program to fail.
        # Change "True" to "False" in file venv/lib/python3.12/site-packages/zeus/device/cpu/rapl.py (l. 137)
        self.measurements = {d["category"]: [] for i, d in self.config["model_zoo"].items()}
        self.scores = {d["category"]: [] for i, d in self.config["model_zoo"].items()}
        self.energy_monitor = ZeusMonitor(gpu_indices=[i for i in range(NUM_GPUS)], approx_instant_energy=True)

        # Algorithm config
        # Q is a virtual queue, i.e., we only keep the sum of all violations, no history.
        self.Q = 0.0

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

    def launch(
        self,
        limit_num_samples: int = None,
        apply_chat_template: bool = False,
        log_samples: bool = False
    ):

        if apply_chat_template:
            logger.warning(
                "Chat template formatting change affects loglikelihood and multiple-choice tasks. See docs/chat-template-readme.md for details."
            )
    
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
            self.run_benchmark(task_output, limit_arg)

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
    ):
        task: Task = task_output.task
        smallest_model_category, smallest_model_instance = next(iter(self.vllm_models.items()))
        smallest_model_instance = smallest_model_instance["vllm_eval_instance"]

        limit = get_sample_size(task, limit_arg)
        # limits.append(limit)
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
    
            for idx, request in enumerate(cloned_reqs): 
                model_resps_mapping = self.run_request(request, timestamp=idx)

    def query_model(self, model, request, model_category):
        self.energy_monitor.begin_window(f"pass_{model_category}")
        output = getattr(
            model["vllm_eval_instance"],
            request.request_type
        )(
            [request] if type(request) is not list else request, disable_tqdm=True
        )

        measurement = self.energy_monitor.end_window(f"pass_{model_category}")

        self.measurements[model_category].append(measurement)
        self.scores[model_category].append(output)

        return output

    def run_request(self, request: Instance, c: int = 3, timestamp: int = 0):
        x_t = self.__sample_from_bernoulli(c=c, timestamp=timestamp)
    
        if x_t == 1: 
            logger.info(f"Exploring during step {timestamp}.")
            
            model_resps_mapping = {i: [] for i in self.vllm_models.keys()}
            for model_category, model in self.vllm_models.items():
                model_resps_mapping[model_category] = self.query_model(
                    model, 
                    request, 
                    model_category
                )
    
            # the final response
            label = []
            for model_category, data in model_resps_mapping.items(): 
                label.append(0 if sum(map(lambda x: x[1], data)) / len(data) < 0.5 else 1)

            label = torch.tensor(label)
            response_idx = label[torch.argmax(label)]
            model_category = [i for i in self.vllm_models.keys()][response_idx]
            response = label[response_idx]
        
            # Prepare the classifier training environment
            # self.__evict_vllm_models()

            # Train the classifier
            classifier_power_measurement = self.__train_classifier_one_step(request=request, label=label.tolist())

            # Need to re-initiate inference models
            # self.__warm_up_inference_models()
    
        else: 
            logger.info(f"Estimating during step {timestamp}.")  

            preference_estimation = self.estimate_user_preferences(request=request)
            preference_estimation = torch.tensor(preference_estimation).reshape(-1, 1)

            energy = []
            for model_category in selector.measurements.keys(): 
                energy.append(
                    sum(map(lambda x: sum([i for i in x.gpu_energy.values()]) / len(selector.measurements[model_category]), selector.measurements[model_category]))
                )
            
            energy = torch.tensor(energy).view(-1, 1)
            # Energy and Preference Estimation are vectors.
            cost_fn = self.algorithm_config["V"] * energy + self.Q * (self.algorithm_config["alpha"] - preference_estimation)
            choice_idx = torch.argmax(cost_fn)
            model_category = [i for i in self.vllm_models.keys()][choice_idx]

            # Query the model of choice
            response = self.query_model(
                self.vllm_models[model_category],
                request, 
                model_category
            )

        return response, model_category

    @staticmethod
    def __sample_from_bernoulli(c: float, timestamp: int):

        p_t = min(
            1.0, c / np.power(1 if timestamp == 0 else timestamp, (1/5))
        )

        x_t = binomial(n=1, p=p_t, size=1)

        return x_t.item()

    def __warm_up_inference_models(self):
        self.vllm_models = {}
        self.tokenizers = {}

        logger.info(f"Found {len(self.config["model_zoo"].keys())} models in zoo: {self.config["model_zoo"].keys()}")
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
                    seed=self.config["seed"]
                )
            }

            logger.info(f"vLLM model {model} loaded on rank {data['gpu_indices']}. Tensor parallel size: {len(data['gpu_indices'])}")

        logger.info(f"All models loaded.")

    def __warmup_classifier_model(self):

        self.classifier_tokenizer = transformers.AutoTokenizer.from_pretrained(self.classifier_config["model_id"])
        base_model = transformers.AutoModel.from_pretrained(self.classifier_config["model_id"])

        trainer_cls = MESSPlusTrainer if self.classifier_config["reweight_classes"] is True else pl.Trainer
        self.classifier_trainer = trainer_cls(
            max_epochs=1,
            accelerator="gpu",
            # logger=WandbLogger(),
            log_every_n_steps=1,
            # callbacks=[checkpointing_callback]
        )

        self.mess_classifier = MESSRouter(
            base_model=base_model,
            model_list=[i for i in self.config["model_zoo"].keys()],
            n_classes=len([i for i in self.config["model_zoo"].keys()]),
            n_epochs=1,
            lr=self.classifier_config["lr"],
            hidden_layer_shape=self.classifier_config["hidden_layer_shape"],
            optim_name=self.classifier_config["optimizer"]
        )

        self.classifier_trainer.configure_criterion(
            kind="bce",
            gamma=self.classifier_config["gamma"] if "gamma" in self.classifier_config.keys() else None
        )

        logger.info(f"Classification model {self.config['classifier_model']['model_id']} loaded and ready to use.")

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

    def __train_classifier_one_step(self, request: Instance, label: List[int]):
        lit_dataloader = MESSOnlineLightningDataloader(
            request=request.doc["passage"] if "passage" in request.doc else request.doc["text"],
            label=label,
            tokenizer=self.classifier_tokenizer,
            seed=self.config["seed"],

        )

        if self.classifier_config["reweight_classes"] is True or self.classifier_config["use_focal_loss"] is True:
            # When training with class-weighted losses, we add the new label to the label history and compute
            # the class weights online. The same goes for Focal Loss.
            self.label_history = pd.concat([self.label_history, pd.DataFrame({"label": str(label)}, index=[0])])
            # self.label_history.reset_index(inplace=True)
            self.classifier_trainer.compute_class_weights(self.label_history)

        self.energy_monitor.begin_window(f"classifier_training_step")
        self.classifier_trainer.fit(self.mess_classifier, datamodule=lit_dataloader)
        measurement = self.energy_monitor.end_window(f"classifier_training_step")

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

    def estimate_user_preferences(self, request: Instance):

        inputs = tokenize_request(request=request.doc["text"], tokenizer=self.classifier_tokenizer)
        inputs = {k: v.to(self.mess_classifier.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mess_classifier(**inputs)

        preds = outputs.logits
        # probs = torch.softmax(preds, dim=-1)

        print(f"PROBS: ", preds)

        # estimated_user_preferences = []
        # for idx in range(len(self.config["model_zoo"].keys())):
        #     # For now, this only works for single requests. When using batched processing, we need to introduce another
        #     # dimension here.
        #     if probs.shape != torch.Size([1, probs.shape[-1]]):
        #         logger.warning(f"It seems you are trying to use batched inputs. Currently the system only supports "
        #                        f"single requests. Change the way probs are summed up.")

        #     estimated_user_preferences.append(
        #         torch.sum(probs[:, 0:(idx + 1)])
        #     )
        #
        # return estimated_user_preferences


if __name__ == "__main__":
    config_file_path = "config/messplus/boolq_baseline.yaml"

    selector = MessPlusAutomaticModelSelector(
        config_file_path=config_file_path
    )

    selector.launch()
