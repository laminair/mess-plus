import argparse
import logging
import numpy as np
import pandas as pd
import wandb
import yaml

from pathlib import Path

from classifier.model import MultilabelBERTClassifier
from classifier.file_reader import read_files_from_folder
from classifier.dataset import create_bert_datasets, preprocess_dataframe

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
SEEDS = [42, 43, 44]
NUM_PRETRAINING_STEPS = 400


def simulate(args):
	for seed in SEEDS:

		set_all_seeds(seed)

		if args.approach == "pretrained":
			config_path = Path(f"{PROJECT_ROOT_PATH}/config/{args.model_family}/pretrained/{args.benchmark_name}.yaml")
			NUM_PRETRAINING_STEPS = args.num_classifier_pretraining_steps
		elif args.approach == "online":
			config_path = Path(f"{PROJECT_ROOT_PATH}/config/{args.model_family}/online/{args.benchmark_name}.yaml")
			NUM_PRETRAINING_STEPS = 0
		else:
			raise NotImplementedError(f"Approach {args.approach} not implemented.")

		with config_path.open("r") as f:
			CONFIG = yaml.safe_load(f)
			logger.info(CONFIG)

		input_df = read_files_from_folder(
			folder_path=f"{PROJECT_ROOT_PATH}/data/{args.model_family}/inference_outputs/{args.benchmark_name}")
		input_df["idx_original"] = input_df.index
		input_df = input_df.sample(frac=1, random_state=seed).reset_index(drop=True)

		logger.info(f"Loaded dataframe with {input_df.shape[0]} rows and {input_df.shape[1]} columns")
		logger.info(f"{len(input_df.columns.tolist())} available columns: {input_df.columns.tolist()}")
		logger.info(input_df.head())

		text_col = "input_text"
		model_labels = [data['category'] for _, data in CONFIG["model_zoo"].items()]
		num_labels = len(model_labels)
		label_cols = [f"label_{label}" for label in model_labels]

		classifier = MultilabelBERTClassifier(num_labels=num_labels, **CONFIG["classifier_model"])

		if args.approach == "pretrained":
			training_df = input_df.loc[:NUM_PRETRAINING_STEPS]
			training_df = preprocess_dataframe(training_df, label_cols=label_cols)

			train_dataset, val_dataset, _ = create_bert_datasets(
				training_df,
				text_col,
				label_cols,
				model_name=CONFIG["classifier_model"]["model_id"],
				max_length=CONFIG["classifier_model"]["max_length"],
				val_ratio=CONFIG["classifier_model"]["validation_dataset_size"],
				random_seed=seed
			)

			training_stats = classifier.fit(
				train_dataset,
				val_dataset,
				epochs=CONFIG["classifier_model"]["epochs"],
				early_stopping_patience=2
			)

			logger.info(training_stats)

		for label in model_labels:
			logger.info(f"{label.upper()} model average accuracy over time: {input_df[NUM_PRETRAINING_STEPS:][f'label_{label}'].mean()}")

		algorithm_config = CONFIG["algorithm"]

		sample_cols = input_df.columns.tolist()

		ALPHA_VALUES = algorithm_config["alpha_values"]
		C_VALUES = [1.0, 0.1, 0.01]
		V_VALUES = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

		for alpha in ALPHA_VALUES:
			for c in C_VALUES:
				for v in V_VALUES:
					algorithm_config["V"] = v
					algorithm_config["alpha"] = alpha
					algorithm_config["c"] = c

					# if seed == 42 and c == 1.0 and v == 1.0:
					# 	logger.info(f"Skipping run for Alpha: {alpha}, C: {c}, V: {v}. Already done before.")
					# 	continue

					ACCURACY_LIST = []
					EXPLORATION_STEP_LIST = []
					ENERGY_CONSUMPTION_LIST = []
					INFERENCE_TIME_LIST = []
					MODEL_CHOSEN_LIST = []
					CLASSIFIER_LOSS_LIST = []
					ENERGY_PER_MODEL = {i: [1] for i in model_labels}

					model_category_list = [i for i in ENERGY_PER_MODEL.keys()]

					Q = 0.0
					ctr = 0

					run_name = (f"{args.benchmark_name}"
					            f"_V={algorithm_config['V']}"
					            f"_a={algorithm_config['alpha']}"
					            f"_c={algorithm_config['c']}"
					            f"_seed={seed}")

					logger.info(f"Starting run for {run_name}")
					with wandb.init(
							entity=args.wandb_entity,
							project=args.wandb_project,
							name=run_name,
							config=CONFIG
					) as run:

						if args.approach == "pretrained":
							run.summary.update({**{f"classifier/{k}": v for k, v in training_stats.items()}})

						run.summary.update({
							"V": algorithm_config["V"],
							"alpha": algorithm_config["alpha"],
							"c": algorithm_config["c"],
							"random_seed": seed
						})

						monitoring_dict = {}
						for idx, sample in input_df[NUM_PRETRAINING_STEPS:].iterrows():
							p_t, x_t = sample_from_bernoulli(c=algorithm_config["c"], timestamp=idx)
							EXPLORATION_STEP_LIST.append(x_t)

							logger.debug(f"Working on sample {idx}")

							if x_t == 1:

								if args.approach == "online":
									training_dataset, _, _ = create_bert_datasets(
										sample,
										text_col,
										label_cols,
										model_name=CONFIG["classifier_model"]["model_id"],
										max_length=CONFIG["classifier_model"]["max_length"],
										val_ratio=CONFIG["classifier_model"]["validation_dataset_size"],
										random_seed=seed
									)

									classifier_metrics_dict = classifier.fit(
										training_dataset,
										epochs=CONFIG["classifier_model"]["epochs"],
										early_stopping_patience=2,
										ctr=ctr,
										online_learn=True
									)

									CLASSIFIER_LOSS_LIST.append(
										classifier_metrics_dict["classifier/step_train_loss"]
									)
									monitoring_dict["classifier/train_loss"] = sum(CLASSIFIER_LOSS_LIST) / (ctr + 1)
									train_energy = classifier_metrics_dict["classifier/train_step_energy"]
									monitoring_dict["classifier/train_step_energy"] = train_energy

								result = sample["label_large"]
								ACCURACY_LIST.append(result)
								step_energy = sum([sample[i] for i in sample_cols if "energy" in i])
								monitoring_dict[f"mess_plus/inference_only_energy"] = step_energy
								if args.approach == "online":
									# We need to add the energy spent on training to the total energy consumption.
									step_energy += train_energy

								step_time = sum([sample[i] for i in sample_cols if "inference" in i])
								chosen_model_id = len(model_category_list) - 1
								MODEL_CHOSEN_LIST.append(chosen_model_id)
								ENERGY_CONSUMPTION_LIST.append(step_energy)
								INFERENCE_TIME_LIST.append(step_time)

								for i in ENERGY_PER_MODEL.keys():
									ENERGY_PER_MODEL[i] = sample[f"energy_consumption_{i}"]

								monitoring_dict[f"mess_plus/total_energy_incl_classifier"] = step_energy
								monitoring_dict[f"mess_plus/chosen_model"] = chosen_model_id


							else:
								preds, probs = classifier.predict(texts=[sample["input_text"]])
								energy = pd.DataFrame(ENERGY_PER_MODEL, index=[0]).to_numpy()

								energy = np.array(energy).reshape(-1, 1)
								probs = probs.reshape(-1, 1)

								cost_fn = algorithm_config["V"] * energy + Q * (alpha - probs)
								cost_fn = cost_fn.reshape(1, -1)
								chosen_model_id = np.argmin(cost_fn)
								model_category_chosen = model_category_list[chosen_model_id]

								result = sample[f"label_{model_category_chosen}"]
								step_energy = sample[f"energy_consumption_{model_category_chosen}"]
								step_time = sample[f"inference_time_{model_category_chosen}"]

								INFERENCE_TIME_LIST.append(step_time)
								ENERGY_CONSUMPTION_LIST.append(step_energy)
								MODEL_CHOSEN_LIST.append(chosen_model_id)
								ACCURACY_LIST.append(result)

								monitoring_dict[f"mess_plus/energy"] = step_energy
								monitoring_dict[f"mess_plus/chosen_model"] = chosen_model_id

							Q = max(0.0, Q + algorithm_config["alpha"] - result)

							x = np.array(MODEL_CHOSEN_LIST)
							monitoring_dict.update({
								"mess_plus/p_t": p_t,
								"mess_plus/x_t": x_t,
								"mess_plus/exploration_step_ratio": sum(EXPLORATION_STEP_LIST) / (ctr + 1),
								"mess_plus/q_length": Q,
								"avg_accuracy": sum(ACCURACY_LIST) / (ctr + 1),
								"step_time": step_time,
								"total_runtime": sum(INFERENCE_TIME_LIST),
								"step_energy_consumption": step_energy,
								# "models/small_chosen": len(np.where(x == 0)[0]) / (len(x) + 1e-8),
								# "models/medium_chosen": len(np.where(x == 1)[0]) / (len(x) + 1e-8),
								# "models/large_chosen": len(np.where(x == 2)[0]) / (len(x) + 1e-8),
							})

							# Add model ids to monitoring dict
							chosen_data = {
								f"models/{label}_chosen": len(np.where(x == ldx)[0]) / (len(x) + 1e-8)
								for ldx, label in enumerate(model_labels)
							}
							monitoring_dict.update(chosen_data)

							ctr += 1
							wandb.log(monitoring_dict, step=idx)

					wandb.finish()
					logger.info(f"Run {run_name} done.")

	logger.info("All simulations done.")


def parse_args():
	parser = argparse.ArgumentParser(description='Train classification model')
	parser.add_argument('--benchmark-name', type=str, required=True,
	                    help='Name of the benchmark you want to run. Must correspond with filename in config folder.')
	parser.add_argument('--model-family', type=str, required=True, choices=["llama3", "qwen2"],
	                    help='Folder name where the config file is located.')
	parser.add_argument('--approach', type=str, required=True, choices=["pretrained", "online"],
	                    help='Whether to use a pre-trained classifier or learn the classifier online')
	parser.add_argument('--wandb-entity', type=str, required=True,
	                    help='W&B entity name')
	parser.add_argument('--wandb-project', type=str, required=True, default="mess-plus_runs_v01",
	                    help='W&B project name')
	parser.add_argument('--num-classifier-pretraining-steps', type=int, required=False, default=400,
	                    help='Number of pre-training steps for the classifier. Only has an effect with approach "pretrained".')
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()

	simulate(args)
