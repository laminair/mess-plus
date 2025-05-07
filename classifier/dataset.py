import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


class BertPandasDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, text_col, y_cols, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.text_col = text_col
        self.y_cols = y_cols
        self.tokenizer = tokenizer
        self.max_length = max_length

        if type(self.dataframe) == pd.DataFrame:
            self.input_type = "df"
        elif type(self.dataframe) == pd.Series:
            self.input_type = "series"
        else:
            raise NotImplementedError("Handler for data type not implemented.")

        # With this we make sure that the iloc function can always find a sample.
        # self.dataframe.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        if self.input_type == "df":
            try:
                text = self.dataframe.iloc[idx][self.text_col].values
            except AttributeError:
                text = self.dataframe.iloc[idx][self.text_col]

            try:
                labels = self.dataframe.iloc[idx][self.y_cols].values
            except AttributeError:
                labels = self.dataframe.iloc[idx][self.y_cols]

        elif self.input_type == "series":
            try:
                text = self.dataframe[self.text_col].values
            except AttributeError:
                text = self.dataframe[self.text_col]

            try:
                labels = self.dataframe[self.y_cols].values
            except AttributeError:
                labels = self.dataframe[self.y_cols]
        else:
            raise NotImplementedError("Handler for data type not implemented.")

        if type(text) is np.ndarray:
            text = text[0]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Explicitly convert the label values to float32
        # This addresses the numpy.object_ conversion error
        labels = labels.astype(np.float32)  # Convert to float32
        item['labels'] = torch.tensor(labels, dtype=torch.float32)

        return item


def create_bert_datasets(df, text_column, label_columns, model_name="bert-base-uncased", max_length=128, val_ratio=0.1,
                         random_seed=42, make_training_dataset_only: bool = False):
    """
    Creates train and validation datasets for BERT models from a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        text_column (str): Column name containing the text to tokenize.
        label_columns (list): List of column names to use as targets.
        model_name (str): Name of the pretrained model to load tokenizer from.
        max_length (int): Maximum sequence length for tokenization.
        val_ratio (float): Fraction of data to use for validation (default: 0.1).
        random_seed (int): Random seed for reproducibility (default: 42).

    Returns:
        tuple: (train_dataset, val_dataset, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create the full dataset
    full_dataset = BertPandasDataset(df, text_column, label_columns, tokenizer, max_length)

    if make_training_dataset_only:
        return full_dataset, None, tokenizer

    else:
        # Calculate sizes for the split
        dataset_size = len(full_dataset)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size

        # Split the dataset
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )

        return train_dataset, val_dataset, tokenizer


# Collate function to handle batching
def collate_fn(batch):
    """Custom collate function for the DataLoader that handles type conversion."""
    if isinstance(batch[0], dict):
        # If the dataset returns dictionaries
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])

        # Make sure labels are all float32 tensors
        labels = []
        for item in batch:
            if isinstance(item['labels'], np.ndarray):
                labels.append(torch.tensor(item['labels'], dtype=torch.float32))
            else:
                labels.append(item['labels'].float())  # Ensure it's float

        labels = torch.stack(labels)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        # Add token_type_ids if present
        if 'token_type_ids' in batch[0]:
            token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
            result['token_type_ids'] = token_type_ids

        return result
    else:
        # If the dataset returns tuples/lists
        raise ValueError("Expected dataset to return dictionaries")


def preprocess_dataframe(df, label_cols):
    """
    Preprocesses the DataFrame to ensure proper types for all columns.

    Args:
        df: pandas DataFrame
        label_cols: list of label column names

    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Ensure label columns are float type
    for col in label_cols:
        # First convert to numeric, coercing errors to NaN
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        # Then fill NaNs with 0
        df_copy[col] = df_copy[col].fillna(0).astype(np.float32)

    return df_copy

