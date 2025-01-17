import os

import pandas as pd


def filter_models_from_dataset(
    df: pd.DataFrame,
    models_to_remove,
    base_model_name: str = "llama",
    limit_nb_samples=None,
    seed: int = 42
):

    df = df[list(set(df.columns.tolist()) - models_to_remove)]
    non_model_cols = [i for i in df.columns.tolist() if base_model_name not in i]
    sorted_model_cols = sorted([i for i in df.columns.tolist() if base_model_name in i])
    df = df[non_model_cols + sorted_model_cols]
    df["label"] = df.apply(lambda row: str([row[i] for i in sorted_model_cols]), axis=1)

    if limit_nb_samples is not None and len(df) > limit_nb_samples:
        df = df.sample(limit_nb_samples, random_state=seed)

    return df
