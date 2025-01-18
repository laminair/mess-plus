import argparse
import logging
import numpy as np
import random
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
from utils.experimentation import filter_dataset
from utils.lit_trainer import WeightedLossTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'mess_plus.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


FILE_PATH = pathlib.Path(__file__).parent
PROJECT_ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()


def train(config=None):

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    pl.seed_everything(SEED, workers=True)

    with wandb.init(config=config) as run:
        config = wandb.config

        run.name = (f"lr_{config.learning_rate}-"
                    f"hls_{config.hidden_layer_shape}-"
                    f"optim_{config.optimizer}-"
                    f"epo_{config.epoch}-"
                    f"mbs_{config.minibatch_size}")

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
            batch_size=config.minibatch_size
        )

        checkpointing_callback = ModelCheckpoint(
            dirpath=f"{FILE_PATH}/checkpoints",
            filename="mess-plus-{epoch}-l{val_loss:.4f}-a{val/accuracy:.4f}",
            monitor="val/loss"
        )

        trainer_cls = WeightedLossTrainer if pipeline_config["reweight_classes"] is True else pl.Trainer
        trainer = trainer_cls(
            max_epochs=config.epoch,
            accelerator="gpu",
            logger=WandbLogger(),
            log_every_n_steps=5,
            callbacks=[checkpointing_callback]
        )

        trainer.test(lit_model, lit_dataloader)

        if pipeline_config["reweight_classes"] is True:
            labels = lit_dataloader.y_train
            trainer.update_class_weights(labels)

        trainer.fit(lit_model, lit_dataloader)
        logger.info(f"The best model can be found under: {checkpointing_callback.best_model_path}.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='MESS+ Classifier Sweeper')
    parser.add_argument("-pc", "--pipeline-config", help="Path to pipeline config (relative to project root)")
    parser.add_argument("-sc", "--sweep-config", help="Path to yaml config (relative to project root)")
    parser.add_argument("-wp", "--wandb-project", help="Project name to use on the W&B console")

    args = parser.parse_args()

    with open(f"{PROJECT_ROOT_PATH}/{args.pipeline_config}", "r") as pipe_cfg:
        pipeline_config = yaml.safe_load(pipe_cfg)
        SEED = pipeline_config["seed"]
        pipe_cfg.close()

    with open(f"{PROJECT_ROOT_PATH}/{args.sweep_config}", "r") as cfg_file:
        sweep_config = yaml.safe_load(cfg_file)
        cfg_file.close()

    df = pd.read_csv(f"{PROJECT_ROOT_PATH}/{pipeline_config['dataset_path']}", low_memory=False)
    df = filter_dataset(
        df,
        benchmark_dataset=pipeline_config["dataset"],
        dataset_name_matching=pipeline_config["dataset_name_matching"],
        base_model_name=pipeline_config["inference_base_model_name"],
        models_to_remove=set(pipeline_config["models_to_remove"]),
        limit_nb_samples=pipeline_config["limit_samples"],
        seed=pipeline_config["seed"]
    )
    inference_models = [i for i in df.columns if "llama_" in i]
    num_inference_models = len(inference_models)

    logger.info(f"Training on {num_inference_models} models: {inference_models}.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(pipeline_config["classifier_base_model"])
    base_model = transformers.AutoModel.from_pretrained(pipeline_config["classifier_base_model"])

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=train)

    api = wandb.Api()
    sweep = api.sweep(path=f"{api.default_entity}/{args.wandb_project}/sweeps/{sweep_id}")
    best_run = sweep.best_run()
    logger.info(f"Best run: {best_run}. Done.")
