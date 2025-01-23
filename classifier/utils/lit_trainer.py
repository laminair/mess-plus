import logging
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .loss_fn import FocalLoss


class MESSPlusTrainer(pl.Trainer):

    def __init__(self, gamma=2.0, alpha=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = gamma
        self.weight = torch.ones((1, 3))
        self.criterion = None
        self.criterion_name = "bce"

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        self.weight = self.weight.to(logits.device)
        loss = self.criterion(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

    def configure_criterion(
        self,
        kind: str = "bce",
        gamma: float = 0.0
    ):
        self.criterion_name = kind
        if kind == "bce" and gamma == 0.0:
            self.criterion = torch.nn.BCELoss(weight=self.weight)
        elif gamma > 0:
            logging.info(f"Using focal loss to train classifier.")
            assert gamma >= 0, "Make sure gamma is configured for focal loss."
            self.criterion_name = "focal_bce"
            self.criterion = FocalLoss(
                alpha=self.weight.squeeze(),
                gamma=gamma
            )
        else:
            raise NotImplementedError("Loss criterion not implemented.")

    def update_class_weights(self, df) -> None:
        """
        When doing online learning we can only update the class weights "as we go".
        :param df: Pandas dataframe with 'label' column containing preference labels the classifier has seen up to
                    timestep t
        :return:
        """
        self.weight = self.compute_class_weights(df)

        if self.criterion_name == "focal_bce":
            self.criterion.update_alpha(self.weight.squeeze())

    @staticmethod
    def compute_class_weights(df) -> torch.Tensor:

        def coerce_dtype(x):
            if type(x) is str:
                return eval(x)

            return x

        df["label"] = df["label"].apply(lambda x: coerce_dtype(x))

        label_matrix = np.stack(df["label"].values, axis=0)
        num_samples, num_labels = label_matrix.shape

        counts = label_matrix.sum(axis=0)  
        freq = counts / (num_samples + 1e-9) 

        weights = 1.0 / (freq + 1e-9)
        weights = weights / weights.mean()  
        weights = weights.reshape(1, num_labels)

        return torch.as_tensor(weights, dtype=torch.float)


class LoggerCallback(pl.Callback):

    def __init__(self, wandb_run):
        self.stats = []

        self.wandb_run = wandb_run

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        logs = {f"classifier/loss": outputs["loss"].cpu().numpy().item()}
        self.wandb_run.log(logs)
