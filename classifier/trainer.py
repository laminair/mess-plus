import logging
import wandb
import yaml
import argparse

from pathlib import Path

from file_reader import read_files_from_folder
from dataset import create_bert_datasets, preprocess_dataframe
from model import MultilabelBERTClassifier


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler(f'mess_plus.log'),
        logging.StreamHandler()
    ]
)

FOLDER_PATH = Path(__file__).parent.absolute()
print(FOLDER_PATH)


def train_model(config):

    logger.info("Starting classifier training")
    benchmark_config_path = Path(args.config_path)

    # Read and parse the YAML file
    with benchmark_config_path.open("r") as f:
        classifier_config = yaml.safe_load(f)["classifier_model"]

    f.close()
    logger.info(f"Configuration loaded from path {args.config_path}")

    if config is not None:
        classifier_config.update(config)
        logger.info(f"Training configuration updated with sweep config: {classifier_config}")


    training_df = read_files_from_folder(args.dataset_path, file_ext=".csv")

    text_col = ["input_text"]
    label_cols = ["label_small", "label_medium", "label_large"]

    dataset = training_df[text_col + label_cols]
    dataset = preprocess_dataframe(dataset, label_cols=label_cols)

    logger.info(f"Dataset loaded. Shape: {dataset.shape} (rows, cols)")

    # Create train and validation datasets
    train_dataset, val_dataset, tokenizer = create_bert_datasets(
        dataset,
        text_col,
        label_cols,
        model_name=classifier_config["model_id"],
        max_length=classifier_config["max_length"],
        val_ratio=classifier_config["validation_dataset_size"]
    )

    logger.info(f"Dataset splits created. Rows - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

    classifier = MultilabelBERTClassifier(
        model_name=classifier_config["model_id"],  # Replace with your preferred BERT variant
        num_labels=len(label_cols),
        learning_rate=classifier_config["learning_rate"],
        momentum=classifier_config["momentum"],
        weight_decay=classifier_config["weight_decay"],
        batch_size=classifier_config["batch_size"],
        max_length=classifier_config["max_length"],
        warmup_ratio=classifier_config["warmup_ratio"],
        threshold=classifier_config["threshold"],
        freeze_bert_layers=classifier_config["freeze_bert_layers"],
        config=classifier_config,
    )

    logger.info(f"Model loaded and ready for training")

    with wandb.init():
        # Train the model
        classifier.fit(train_dataset, val_dataset, epochs=classifier_config["epochs"], early_stopping_patience=2)

    wandb.finish()
    logger.info("Classifier training done.")


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification model')
    parser.add_argument('--config-path', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--wandb-entity', type=str, required=True,
                        help='W&B entity name')
    parser.add_argument('--wandb-project', type=str, required=True,
                        help='W&B project name')
    parser.add_argument('--wandb-run', type=str, required=True,
                        help='W&B run name')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to dataset')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(f"{FOLDER_PATH}/sweep_config.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)

    f.close()

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    wandb.agent(
        sweep_id,
        train_model,
        count=5
    )
