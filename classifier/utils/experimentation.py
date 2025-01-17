import os


def filter_models_from_dataset(df, models_to_remove, base_model_name: str = "llama"):
    df = df[list(set(df.columns.tolist()) - models_to_remove)]
    non_model_cols = [i for i in df.columns.tolist() if base_model_name not in i]
    sorted_model_cols = sorted([i for i in df.columns.tolist() if base_model_name in i])
    df = df[non_model_cols + sorted_model_cols]
    df["label"] = df.apply(lambda row: str([row[i] for i in sorted_model_cols]), axis=1)

    return df
