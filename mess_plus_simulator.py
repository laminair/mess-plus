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

# ------------------------------------------------------------------
# 🔄 SWITCH METRIC → API COST  (Together.ai June-2025 price sheet)
#    – 1 B  = $0.06  / 1 M tokens
#    – 8 B  = $0.18  / 1 M tokens
#    – 70 B = $0.88  / 1 M tokens
# ------------------------------------------------------------------
_PRICE_MILLION = {"1b": 0.06, "8b": 0.18, "70b": 0.88}
PRICE_PER_TOKEN = {k: v / 1_000_000 for k, v in _PRICE_MILLION.items()}
MODEL_SIZE_MAP = {"small": "1b", "medium": "8b", "large": "70b"}
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("mess_plus.log"),  
        logging.StreamHandler(),
    ],
)

PROJECT_ROOT_PATH = Path(__file__).parent
SEEDS = [42, 43, 44]
NUM_PRETRAINING_STEPS = 400
NUM_CLASSIFIER_LABELS = 3


def simulate(args):

    for seed in SEEDS:

        set_all_seeds(seed)

        if args.approach == "pretrained":
            config_path = PROJECT_ROOT_PATH / "config" / "pretrained" / f"{args.benchmark_name}.yaml"
            NUM_PRETRAINING_STEPS = args.num_classifier_pretraining_steps
        elif args.approach == "online":
            config_path = PROJECT_ROOT_PATH / "config" / "online" / f"{args.benchmark_name}.yaml"
            NUM_PRETRAINING_STEPS = 0
        else:
            raise NotImplementedError(f"Approach {args.approach} not implemented.")

        with config_path.open("r") as f:
            CONFIG = yaml.safe_load(f)
            logger.info(CONFIG)

        input_df = read_files_from_folder(
            folder_path=PROJECT_ROOT_PATH / "data" / "inference_outputs" / args.benchmark_name
        )
        input_df["idx_original"] = input_df.index
        input_df = input_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        logger.info(
            f"Loaded dataframe with {input_df.shape[0]} rows and {input_df.shape[1]} columns"
        )
        logger.info(f"{len(input_df.columns.tolist())} available columns: {input_df.columns.tolist()}")
        logger.info(input_df.head())

        text_col = "input_text"
        label_cols = ["label_small", "label_medium", "label_large"]

        classifier = MultilabelBERTClassifier(
            num_labels=NUM_CLASSIFIER_LABELS, **CONFIG["classifier_model"]
        )

        # -----------------------------------------------------------
        # ❶ (optional) PRETRAIN classifier on first NUM_PRETRAINING_STEPS rows
        # -----------------------------------------------------------
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
                random_seed=seed,
            )

            training_stats = classifier.fit(
                train_dataset,
                val_dataset,
                epochs=CONFIG["classifier_model"]["epochs"],
                early_stopping_patience=2,
            )

            logger.info(training_stats)

        logger.info(
            f"Small model average accuracy over time: "
            f"{input_df[NUM_PRETRAINING_STEPS:]['label_small'].mean()}"
        )
        logger.info(
            f"Medium model average accuracy over time: "
            f"{input_df[NUM_PRETRAINING_STEPS:]['label_medium'].mean()}"
        )
        logger.info(
            f"Large model average accuracy over time: "
            f"{input_df[NUM_PRETRAINING_STEPS:]['label_large'].mean()}"
        )

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

                    # Skip already-known baseline run to save time
                    if seed == 42 and c == 1.0 and v == 1.0:
                        logger.info(f"Skipping run for Alpha: {alpha}, C: {c}, V: {v}. Already done before.")
                        continue

                    ACCURACY_LIST = []
                    EXPLORATION_STEP_LIST = []
                    # ENERGY_CONSUMPTION_LIST = []   # (energy kept but disabled)
                    INFERENCE_TIME_LIST = []
                    MODEL_CHOSEN_LIST = []
                    CLASSIFIER_LOSS_LIST = []
                    COST_LIST = []  # new list for step-level USD costs

                    # Keeps latest energy per size (still useful for cost_fn formula)
                    ENERGY_PER_MODEL = {"small": [1], "medium": [1], "large": [1]}
                    model_category_list = list(ENERGY_PER_MODEL.keys())

                    Q = 0.0
                    ctr = 0

                    run_name = (
                        f"{args.benchmark_name}"
                        f"_V={algorithm_config['V']}"
                        f"_a={algorithm_config['alpha']}"
                        f"_c={algorithm_config['c']}"
                        f"_seed={seed}"
                        f"_together.ai"
                    )


                    logger.info(f"Starting run for {run_name}")
                    with wandb.init(
                        entity=args.wandb_entity,
                        project=args.wandb_project,
                        name=run_name,
                        config=CONFIG,
                    ) as run:

                        if args.approach == "pretrained":
                            run.summary.update(
                                {f"classifier/{k}": v for k, v in training_stats.items()}
                            )

                        run.summary.update(
                            {
                                "V": algorithm_config["V"],
                                "alpha": algorithm_config["alpha"],
                                "c": algorithm_config["c"],
                                "random_seed": seed,
                                "pricing_scheme": "TogetherAI_June2025",
                            }
                        )

                        monitoring_dict = {}
                        for idx, sample in input_df[NUM_PRETRAINING_STEPS:].iterrows():
                            p_t, x_t = sample_from_bernoulli(c=algorithm_config["c"], timestamp=idx)
                            EXPLORATION_STEP_LIST.append(x_t)

                            logger.debug(f"Working on sample {idx}")

                            # --------------------------------------------------
                            # ❷  EXPLORATION  (x_t == 1)  → always use LARGE model
                            # --------------------------------------------------
                            if x_t == 1:

                                if args.approach == "online":
                                    training_dataset, _, _ = create_bert_datasets(
                                        sample,
                                        text_col,
                                        label_cols,
                                        model_name=CONFIG["classifier_model"]["model_id"],
                                        max_length=CONFIG["classifier_model"]["max_length"],
                                        val_ratio=CONFIG["classifier_model"]["validation_dataset_size"],
                                        random_seed=seed,
                                    )

                                    classifier_metrics_dict = classifier.fit(
                                        training_dataset,
                                        epochs=CONFIG["classifier_model"]["epochs"],
                                        early_stopping_patience=2,
                                        ctr=ctr,
                                        online_learn=True,
                                    )

                                    CLASSIFIER_LOSS_LIST.append(
                                        classifier_metrics_dict["classifier/step_train_loss"]
                                    )
                                    monitoring_dict["classifier/train_loss"] = sum(CLASSIFIER_LOSS_LIST) / (
                                        ctr + 1
                                    )
                                    # train_energy = classifier_metrics_dict["classifier/train_step_energy"]
                                    # monitoring_dict["classifier/train_step_energy"] = train_energy

                                result = sample["label_large"]
                                ACCURACY_LIST.append(result)

                                # step_energy = sum(sample[i] for i in sample_cols if "energy" in i)
                                # monitoring_dict["mess_plus/inference_only_energy"] = step_energy
                                # if args.approach == "online":
                                #     step_energy += train_energy

                                step_time = sum(sample[i] for i in sample_cols if "inference" in i)
                                chosen_model_id = len(model_category_list) - 1  # index of "large"
                                MODEL_CHOSEN_LIST.append(chosen_model_id)
                                # ENERGY_CONSUMPTION_LIST.append(step_energy)
                                INFERENCE_TIME_LIST.append(step_time)

                                # Update ENERGY_PER_MODEL cache (still used by cost_fn)
                                for i in ENERGY_PER_MODEL.keys():
                                    ENERGY_PER_MODEL[i] = sample[f"energy_consumption_{i}"]

                                # monitoring_dict["mess_plus/total_energy_incl_classifier"] = step_energy
                                monitoring_dict["mess_plus/chosen_model"] = chosen_model_id

                                # ---- cost for LARGE model ----
                                price_key = MODEL_SIZE_MAP["large"]
                                step_cost = PRICE_PER_TOKEN[price_key] * sample["tokens_large"]
                                COST_LIST.append(step_cost)
                                monitoring_dict["mess_plus/step_cost_usd"] = step_cost

                            # --------------------------------------------------
                            # ❸  EXPLOITATION  (x_t == 0)  → choose via cost_fn
                            # --------------------------------------------------
                            else:
                                preds, probs = classifier.predict(texts=[sample["input_text"]])
                                energy_vector = (
                                    pd.DataFrame(ENERGY_PER_MODEL, index=[0]).to_numpy().reshape(-1, 1)
                                )
                                probs = probs.reshape(-1, 1)

                                cost_fn = algorithm_config["V"] * energy_vector + Q * (
                                    alpha - probs
                                )
                                chosen_model_id = int(np.argmin(cost_fn))
                                model_category_chosen = model_category_list[chosen_model_id]

                                result = sample[f"label_{model_category_chosen}"]
                                # step_energy = sample[f"energy_consumption_{model_category_chosen}"]
                                step_time = sample[f"inference_time_{model_category_chosen}"]

                                INFERENCE_TIME_LIST.append(step_time)
                                # ENERGY_CONSUMPTION_LIST.append(step_energy)
                                MODEL_CHOSEN_LIST.append(chosen_model_id)
                                ACCURACY_LIST.append(result)

                                # monitoring_dict["mess_plus/energy"] = step_energy
                                monitoring_dict["mess_plus/chosen_model"] = chosen_model_id

                                # ---- cost for chosen model ----
                                price_key = MODEL_SIZE_MAP[model_category_chosen]
                                token_col = f"tokens_{model_category_chosen}"
                                step_cost = PRICE_PER_TOKEN[price_key] * sample[token_col]
                                COST_LIST.append(step_cost)
                                monitoring_dict["mess_plus/step_cost_usd"] = step_cost

                            # ---------- backlog update ----------
                            Q = max(0.0, Q + algorithm_config["alpha"] - result)

                            x_arr = np.array(MODEL_CHOSEN_LIST)
                            monitoring_dict.update(
                                {
                                    "mess_plus/p_t": p_t,
                                    "mess_plus/x_t": x_t,
                                    "mess_plus/exploration_step_ratio": sum(EXPLORATION_STEP_LIST)
                                    / (ctr + 1),
                                    "mess_plus/q_length": Q,
                                    "avg_accuracy": sum(ACCURACY_LIST) / (ctr + 1),
                                    "step_time": step_time,
                                    "total_runtime": sum(INFERENCE_TIME_LIST),
                                    # "step_energy_consumption": step_energy,          # disabled
                                    "step_cost_usd": step_cost,
                                    "running_avg_cost_usd": sum(COST_LIST) / (ctr + 1),
                                    "running_total_cost_usd": sum(COST_LIST),
                                    "models/small_chosen": len(np.where(x_arr == 0)[0])
                                    / (len(x_arr) + 1e-8),
                                    "models/medium_chosen": len(np.where(x_arr == 1)[0])
                                    / (len(x_arr) + 1e-8),
                                    "models/large_chosen": len(np.where(x_arr == 2)[0])
                                    / (len(x_arr) + 1e-8),
                                }
                            )

                            ctr += 1
                            wandb.log(monitoring_dict, step=idx)

                    wandb.finish()
                    logger.info(f"Run {run_name} done.")

    logger.info("All simulations done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train classification model")
    parser.add_argument(
        "--benchmark-name",
        type=str,
        required=True,
        help="Name of the benchmark you want to run. Must correspond with filename in config folder.",
    )
    parser.add_argument(
        "--approach",
        type=str,
        required=True,
        choices=["pretrained", "online"],
        help="Whether to use a pre-trained classifier or learn the classifier online",
    )
    parser.add_argument("--wandb-entity", type=str, required=True, help="W&B entity name")
    parser.add_argument(
        "--wandb-project",
        type=str,
        required=True,
        default="mess-plus_runs_v01",
        help="W&B project name",
    )
    parser.add_argument(
        "--num-classifier-pretraining-steps",
        type=int,
        required=False,
        default=400,
        help='Number of pre-training steps for the classifier. Only has an effect with approach "pretrained".',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    simulate(args)
