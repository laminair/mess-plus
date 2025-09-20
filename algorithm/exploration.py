import pandas as pd

from classifier.dataset import create_bert_datasets

from typing import Dict, List


def explore(
        sample: pd.DataFrame,
        classifier,
        classifier_config: Dict,
        approach: str = "online",
        text_col: str = "input_text",
        seed: int = 42,
        step: int = 1,

    ):

    label_cols = [col for col in sample.columns if "label_" in col]

    classifier_metrics_dict = {}
    if approach == "online":
        training_dataset, _, _ = create_bert_datasets(
            sample,
            text_col,
            label_cols,
            model_name=classifier_config["model_id"],
            max_length=classifier_config["max_length"],
            val_ratio=classifier_config["validation_dataset_size"],
            random_seed=seed
        )

        classifier_metrics_dict = classifier.fit(
            training_dataset,
            epochs=classifier_config["epochs"],
            early_stopping_patience=2,
            ctr=step,
            online_learn=True
        )

    return classifier_metrics_dict





