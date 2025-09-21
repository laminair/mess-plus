import argparse
import logging
import numpy as np
import pandas as pd
import random
import wandb
import yaml

from pathlib import Path

from classifier.model import MultilabelBERTClassifier
from classifier.file_reader import read_files_from_folder
from classifier.dataset import create_bert_datasets, preprocess_dataframe

from algorithm import explore, exploit, update_Q

from utils.mess_plus import sample_from_bernoulli
from utils.misc import set_all_seeds

logger = logging.getLogger(__name__)
logging.basicConfig(
	level=logging.DEBUG,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		# logging.FileHandler(f'mess_plus.log'),
		logging.StreamHandler()
	]
)

PROJECT_ROOT_PATH = Path(__file__).parent


class MessPlusAutomaticModelSelectorSimulator:

    def __init__(self, args):
        self.config = yaml.safe_load(open(args.config, "r"))
        self.lm_eval_config = self.config["lm_eval"]
        self.algorithm_config = self.config["simulated"]

        self.wandb_project_name = args.wandb_project
        self.wandb_entity = args.wandb_entity

        # Classifier model
        self.classifier_config = self.config["classifier_model"]
        self.labels = [data["category"] for data in self.config["model_zoo"].values()]
        self.text_col = "input_text"

        self.dataset = read_files_from_folder(
			folder_path=args.dataset_path
        )
        self.dataset["idx_original"] = self.dataset.index

        logger.info(f"Loaded dataframe with {self.dataset.shape[0]} rows and {self.dataset.shape[1]} columns")
        logger.info(f"{len(self.dataset.columns.tolist())} available columns: {self.dataset.columns.tolist()}")
        logger.info(self.dataset.head())


    def launch(self):

        if type(self.algorithm_config["seed"]) is int:
            self.algorithm_config["seed"] = [self.algorithm_config["seed"]]

        for seed in self.algorithm_config["seed"]:
            set_all_seeds(seed)
            self.dataset = self.dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

            if self.classifier_config["approach"] == "pretrained":
                # Note: When you choose 'pretrained' the predictor still gets trained in the exploration phase!
                classifier = MultilabelBERTClassifier(num_labels=len(self.labels), **self.config["classifier_model"])
                classifier.load_model(self.classifier_config["checkpoint_path"])
            else:
                classifier = MultilabelBERTClassifier(num_labels=len(self.labels), **self.config["classifier_model"])
                classifier.make_model_if_not_exists(num_labels=len(self.labels))


            for label in self.labels:
                logger.info(f"{label.upper()} model average accuracy over time: {self.dataset[f'label_{label}'].mean()}")

            sample_cols = self.dataset.columns.tolist()

            for alpha in self.algorithm_config["alpha_values"]:
                for c in self.algorithm_config["c_values"]:
                    for v in self.algorithm_config["v_values"]:

                        ACCURACY_LIST = []
                        EXPLORATION_STEP_LIST = []
                        INFERENCE_TIME_LIST = []
                        MODEL_CHOSEN_LIST = []
                        CLASSIFIER_LOSS_LIST = []
                        ENERGY_PER_MODEL = {i: [1] for i in self.labels}

                        model_category_list = [i for i in ENERGY_PER_MODEL.keys()]

                        Q = 0.0
                        Q_tracked = 0.0
                        ctr = 0

                        # Setup the classifier
                        if self.classifier_config["approach"] != "pretrained":
                            classifier = MultilabelBERTClassifier(num_labels=len(self.labels), **self.classifier_config)
                            classifier.make_model_if_not_exists(num_labels=len(self.labels))

                        run_name = (f"{self.config['benchmark']}"
                                    f"_V={v}"
                                    f"_a={alpha}"
                                    f"_c={c}"
                                    f"_seed={seed}")

                        logger.info(f"Starting run for {run_name}")
                        with wandb.init(
                                entity=args.wandb_entity,
                                project=args.wandb_project,
                                name=run_name,
                                config=self.config
                        ) as run:

                            run.summary.update({
                                "V": v,
                                "alpha": alpha,
                                "c": c,
                                "random_seed": seed
                            })

                            monitoring_dict = {}
                            label_cols = [col for col in self.dataset.columns if "label_" in col]
                            for timestamp, sample in self.dataset.iterrows():
                                p_t, x_t = sample_from_bernoulli(c=c, timestamp=timestamp)

                                sample_has_feedback = True
                                if "feedback_sparsity" in self.algorithm_config.keys() and self.algorithm_config["feedback_sparsity"] != 1:
                                    sample_has_feedback = random.uniform(0, 1) <= self.algorithm_config["feedback_sparsity"]

                                EXPLORATION_STEP_LIST.append(x_t)

                                logger.debug(f"Working on sample {timestamp}")

                                if x_t == 1:
                                    logger.info(f"Exploring for step {timestamp}")
                                    exploration_metrics = explore(
                                        sample=sample,
                                        classifier=classifier,
                                        classifier_config=self.classifier_config,
                                        text_col=self.text_col,
                                        approach=self.classifier_config["approach"],
                                        seed=seed,
                                        step=timestamp,
                                        label_cols=label_cols
                                    )

                                    if "classifier/step_train_loss" in exploration_metrics.keys():
                                        CLASSIFIER_LOSS_LIST.append(
                                            exploration_metrics["classifier/step_train_loss"]
                                        )
                                        monitoring_dict["classifier/train_loss"] = sum(CLASSIFIER_LOSS_LIST) / (ctr + 1)

                                    train_energy = 0
                                    if "classifier/train_step_energy" in exploration_metrics.keys():
                                        CLASSIFIER_LOSS_LIST.append(
                                            exploration_metrics["classifier/train_step_energy"]
                                        )
                                        train_energy = exploration_metrics["classifier/train_step_energy"]

                                    monitoring_dict["classifier/train_step_energy"] = train_energy

                                    target_metric = sample[label_cols[-1]]
                                    ACCURACY_LIST.append(target_metric)

                                    step_energy = sum([sample[i] for i in sample_cols if "energy" in i])
                                    monitoring_dict[f"mess_plus/inference_only_energy"] = step_energy
                                    if self.classifier_config["approach"] == "online":
                                        # We need to add the energy spent on training to the total energy consumption.
                                        step_energy += train_energy

                                    step_time = sum([sample[i] for i in sample_cols if "inference" in i])
                                    chosen_model_id = len(model_category_list) - 1
                                    MODEL_CHOSEN_LIST.append(chosen_model_id)
                                    INFERENCE_TIME_LIST.append(step_time)

                                    for i in ENERGY_PER_MODEL.keys():
                                        ENERGY_PER_MODEL[i] = sample[f"energy_consumption_{i}"]

                                    monitoring_dict[f"mess_plus/total_energy_incl_classifier"] = step_energy
                                    monitoring_dict[f"mess_plus/chosen_model"] = chosen_model_id

                                else:
                                    logger.info(f"Exploiting for step {timestamp}")

                                    model_category_chosen = exploit(
                                        classifier=classifier,
                                        input_text=sample[self.text_col],
                                        energy_history=ENERGY_PER_MODEL,
                                        V=v,
                                        alpha=alpha,
                                        Q_tracked=Q_tracked,
                                        label_cols=label_cols
                                    )

                                    chosen_model_id = self.labels.index(model_category_chosen)
                                    print(f"MODEL ID: {chosen_model_id}")
                                    target_metric = sample[f"label_{model_category_chosen}"]
                                    step_time = sample[f"inference_time_{model_category_chosen}"]
                                    step_energy = sample[f"energy_consumption_{model_category_chosen}"]

                                    INFERENCE_TIME_LIST.append(step_time)
                                    MODEL_CHOSEN_LIST.append(chosen_model_id)
                                    ACCURACY_LIST.append(target_metric)

                                    monitoring_dict[f"mess_plus/energy"] = step_energy
                                    monitoring_dict[f"mess_plus/chosen_model"] = chosen_model_id

                                # We are tracking Q twice:
                                #   - Q just monitors the perfect state, i.e., where the Q length should be without sparse user feedback.
                                #   - Q_tracked is used to actually inform the decision-making process (under varying conditions)
                                Q, Q_tracked = update_Q(
                                    Q=Q,
                                    Q_tracked=Q_tracked,
                                    alpha=alpha,
                                    user_satisfaction=target_metric,
                                    sample_has_feedback=sample_has_feedback

                                )

                                x = np.array(MODEL_CHOSEN_LIST)
                                monitoring_dict.update({
                                    "mess_plus/p_t": p_t,
                                    "mess_plus/x_t": x_t,
                                    "mess_plus/exploration_step_ratio": sum(EXPLORATION_STEP_LIST) / (ctr + 1),
                                    "mess_plus/q_optimal_length": Q,
                                    "mess_plus/q_tracked_length": Q_tracked,
                                    # This is what you should look at when interested in Q dynamics
                                    "avg_accuracy": sum(ACCURACY_LIST) / (ctr + 1),
                                    "step_time": step_time,
                                    "total_runtime": sum(INFERENCE_TIME_LIST),
                                    "step_energy_consumption": step_energy
                                })

                                # Add model ids to monitoring dict
                                chosen_data = {
                                    f"models/{label}_chosen": len(np.where(x == ldx)[0]) / (len(x) + 1e-8)
                                    for ldx, label in enumerate(self.labels)
                                }
                                monitoring_dict.update(chosen_data)

                                ctr += 1
                                wandb.log(monitoring_dict, step=timestamp)

                        wandb.finish()
                        logger.info(f"Run {run_name} done.")

        logger.info("All simulations done.")


def parse_args():
    parser = argparse.ArgumentParser(description='MESS+ Algorithm Executor with LM-Eval integration')
    parser.add_argument('--config', type=str, required=True,
                        help='Name of the benchmark you want to run. Must correspond with filename in config folder.')
    parser.add_argument('--wandb-entity', type=str, required=True,
                        help='W&B entity name')
    parser.add_argument('--wandb-project', type=str, required=True, default="messplus_test",
                        help='W&B project name')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to a CSV dataset that contains input_text, LLM labels, and LLM energy consumption.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    simulator = MessPlusAutomaticModelSelectorSimulator(args)
    simulator.launch()
