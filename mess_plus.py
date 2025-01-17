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

from utils.modelling_messplus_classifier import make_mlp

from zeus.monitor import ZeusMonitor

from typing import List, Tuple

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

    def run_dataset(self, dataset: pd.DataFrame, text_col: str, label_col: str) -> None:
        """
        This function expects a pandas dataframe where there is a text input and a corresponding label.
        :param dataset: pandas DataFrame containing the dataset.
        :param text_col: Name of the text column
        :param label_col: Name of the label column
        :return: None
        """

        df = dataset[[text_col, label_col]]
        df.reset_index(inplace=True)
        # We need the index to start at 1.
        df.index += 1

        for idx, row in df.iterrows():

            print(row[text_col])
            print(row[label_col])

            self.run_request(
                req=row[text_col],
                ground_truth=row[label_col],
                timestamp=idx,
                c=self.config["algorithm"]["c"]
            )

            if idx % 10 == 0:
                logger.info(f"Processed indices {idx - 10} to {idx}.")

        logger.info("Dataset processing complete.")

    def run_request(self, req: str, ground_truth: str, timestamp: int, c: float):
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
            return responses[-1]

        else:
            # Let's estimate
            preferences = self.estimate_user_preferences(req=req)
            print(preferences)

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
                "vllm_instance": LLM(
                    model,
                    max_model_len=data["max_seq_len"],
                    trust_remote_code=True,
                    tensor_parallel_size=len(data["gpu_indices"]),
                    gpu_memory_utilization=max_memory_utilization_per_model
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

        trainer = Trainer(
            model=self.classifier_model,
            args=training_args,
            train_dataset=data,
            eval_dataset=data,
            compute_metrics=self.compute_metrics,
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
