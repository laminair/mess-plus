import pandas as pd
import pytorch_lightning as pl
import re
import torch

from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def pre_process(text):
    text = BeautifulSoup(text).get_text()
    # fetch alphabetic characters
    text = re.sub("[^a-zA-Z]", " ", text)
    # convert text to lower case
    text = text.lower()
    # split text into words to remove whitespaces
    tokens = text.split()
    return " ".join(tokens)


def load_data(df, seed, test_val_dataset_size: float = 0.15):
    x_train, x_test, y_train, y_test = train_test_split(
        df["input_text"],
        df["label"],
        test_size=test_val_dataset_size,
        random_state=seed,
        shuffle=True
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=test_val_dataset_size,
        random_state=seed,
        shuffle=False
    )

    x_train = x_train.tolist()
    y_train = y_train.tolist()

    x_val = x_val.tolist()
    y_val = y_val.tolist()

    x_test = x_test.tolist()
    y_test = y_test.tolist()

    # Convert string elements to list
    y_train = [eval(i) for i in y_train]
    y_val = [eval(i) for i in y_val]
    y_test = [eval(i) for i in y_test]

    return x_train, x_val, x_test, y_train, y_val, y_test


class ClassificationDataset(Dataset):

    def __init__(self, x: list, y: list, tokenizer):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(
            self.x[idx],
            None,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "label": torch.tensor(self.y[idx], dtype=torch.float)
        }


class MESSLightningDataloader(pl.LightningDataModule):

    def __init__(self, df: pd.DataFrame, tokenizer, batch_size: int = 64, seed: int = 42):
        super().__init__()
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = load_data(
            df=df,
            seed=seed
        )

        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.trainset = ClassificationDataset(x=self.x_train, y=self.y_train, tokenizer=self.tokenizer)
        self.valset = ClassificationDataset(x=self.x_val, y=self.y_val, tokenizer=self.tokenizer)
        self.testset = ClassificationDataset(x=self.x_test, y=self.y_test, tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)
