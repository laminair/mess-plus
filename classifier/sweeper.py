import argparse
import logging

import pytorch_lightning as pl
import pathlib
import pandas as pd
import transformers
import wandb
import yaml

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.data_management import MESSLightningDataloader
from utils.modelling_mess_plus_classifier import MESSRouter


FILE_PATH = pathlib.Path(__file__).parent
PROJECT_ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()

DATASET_PATH = "datasets/csv/all_data.csv"
DATASET_NAME = "boolq"
MODEL_NAME = "answerdotai/ModernBERT-base"


def train(config=None):

    with wandb.init(config=config):
        config = wandb.config
        print(config)
        lit_model = MESSRouter(
            base_model=base_model,
            model_list=inference_models,
            n_classes=num_inference_models,
            n_epochs=config.epoch,
            lr=config.learning_rate,
            hidden_layer_shape=config.hidden_layer_shape,
            optim_name=config.optimizer
        )

        lit_dataloader = MESSLightningDataloader(
            df=df,
            tokenizer=tokenizer,
            batch_size=config.minibatch_size,
            seed=42
        )

        checkpointing_callback = ModelCheckpoint(
            dirpath=f"{FILE_PATH}/checkpoints",
            filename="mess-plus-{epoch}-l{val_loss:.4f}-a{val/accuracy:.4f}",
            monitor="val/loss"
        )

        trainer = pl.Trainer(
            max_epochs=config.epoch,
            accelerator="gpu",
            logger=WandbLogger(),
            log_every_n_steps=5,
            callbacks=[checkpointing_callback]
        )

        trainer.fit(lit_model, lit_dataloader)
        logging.info(f"The best model can be found under: {checkpointing_callback.best_model_path}.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='MESS+ Classifier Sweeper')
    parser.add_argument("-c", "--config", help="Path to yaml config (relative to project root)")

    args = parser.parse_args()

    with open(f"{PROJECT_ROOT_PATH}/{args.config}", "r") as cfg_file:
        sweep_config = yaml.safe_load(cfg_file)
        cfg_file.close()

    df = pd.read_csv(f"{PROJECT_ROOT_PATH}/{DATASET_PATH}", low_memory=False)
    df = df.loc[df["dataset"] == DATASET_NAME]
    inference_models = [i for i in df.columns if "llama_" in i]
    num_inference_models = len(inference_models)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = transformers.AutoModel.from_pretrained(MODEL_NAME)

    sweep_id = wandb.sweep(sweep_config, project='mess-plus-classifier-boolq-sweep-v1')
    wandb.agent(sweep_id, function=train)
