import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from torch import nn

from .metrics import compute_scores
from .loss_fn import FocalLoss


def mean_pooling(model_output, attention_mask):
    # We need to make sure we only pool actual tokens and not padding.
    # Please see for reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class MESSPlusMLP(nn.Module):
    def __init__(self, hidden_size, num_labels: int, hidden_dim_shapes: list):
        super().__init__()

        layer_dims: list = hidden_dim_shapes
        layer_dims.insert(0, hidden_size)
        layer_dims.append(num_labels)

        self.mlp = nn.ModuleList([
            nn.Linear(hidden_size, layer_dims[1]),
            nn.ReLU()
        ])

        for idx, layer_dim in enumerate(layer_dims[2:]):
            linear_layers = [i for i in self.mlp if hasattr(i, "out_features")]
            self.mlp.append(
                nn.Linear(linear_layers[idx].out_features, layer_dim),
            )

            if not layer_dim == num_labels:
                self.mlp.append(
                    nn.ReLU()
                )

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)


class MESSRouter(pl.LightningModule):

    def __init__(
        self,
        base_model,
        model_list: list,
        hidden_layer_shape: list,
        optim_name: str,
        n_classes: int = 3,
        n_epochs: int = 3,
        lr: float = 0.0001
    ):
        super().__init__()

        self.backbone = base_model

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = MESSPlusMLP(
            self.backbone.config.hidden_size,
            n_classes,
            hidden_dim_shapes=hidden_layer_shape
        )

        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCELoss()

        self.model_list = model_list
        self.optim_name = optim_name

        # Metrics
        self.metrics_df = pd.DataFrame()
        self.val_losses = []

        # We ignore the base_model since it's a pre-trained component.
        self.save_hyperparameters(ignore=['base_model'])

    def forward(self, input_ids, attn_mask):

        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attn_mask)

        # The output here is shaped as follows: (BATCH_SIZE, 768)
        out = mean_pooling(out.last_hidden_state, attn_mask)
        out = self.classifier(out)
        out = torch.sigmoid(out)

        return out

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        probs = self(input_ids, attention_mask)

        loss = self.criterion(probs, labels)
        self.log('train/loss', loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": probs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self):
        self._on_epoch_end_shared(stage="val")

    def test_step(self, batch, batch_idx):
        # The stage is intentionally set to "val" to capture the model accuracy before starting the training.
        self._step(batch, batch_idx, stage="val")

    def on_test_epoch_end(self) -> None:
        # The stage is intentionally set to "val" to capture the model accuracy before starting the training.
        self._on_epoch_end_shared(stage="val")

    def configure_optimizers(self):
        if self.optim_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optim_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Optimizer not configured.")

        return optimizer

    def _step(self, batch, batch_idx, stage):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        probs = self(input_ids, attention_mask)
        loss = self.criterion(probs, labels)

        metrics, conf_mx = compute_scores(
            probs,
            labels,
            stage=stage,
            include_confusion_matrix=False
        )

        metrics.update({
            "batch_idx": batch_idx,
            "stage": stage
        })

        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame(metrics, index=[0])])

        self.val_losses.append(loss)
        self.log(f"{stage}_loss", loss, prog_bar=True, logger=False)
        return loss

    def _on_epoch_end_shared(self, stage: str):

        numeric_cols = [i for i in self.metrics_df.columns.tolist() if stage in i]
        epoch_metrics = self.metrics_df.loc[self.metrics_df["stage"] == stage, numeric_cols].mean()
        epoch_metrics = epoch_metrics.to_dict()
        epoch_metrics[f"{stage}/loss"] = np.mean([i.cpu().item() for i in self.val_losses])

        logging.info(epoch_metrics)
        self.log_dict(epoch_metrics)

        self.metrics_df = pd.DataFrame()
        self.val_losses = []


class MESSRouterNoLightning(nn.Module):

    def __init__(
        self,
        base_model,
        model_list: list,
        hidden_layer_shape: list,
        n_classes: int = 3,
        n_epochs: int = 3,
        gamma: int = 2
    ):
        super().__init__()
        self.focal_gamma = gamma
        self.weight = torch.ones((1, 3))
        self.criterion = None
        self.criterion_name = "bce"

        self.backbone = base_model

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = MESSPlusMLP(
            self.backbone.config.hidden_size,
            n_classes,
            hidden_dim_shapes=hidden_layer_shape
        )

        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()

        self.model_list = model_list

        # Metrics
        self.metrics_df = pd.DataFrame()
        self.val_losses = []

    def forward(self, input_ids, attn_mask):
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attn_mask)

        # The output here is shaped as follows: (BATCH_SIZE, 768)
        out = mean_pooling(out.last_hidden_state, attn_mask)
        out = self.classifier(out)
        out = torch.sigmoid(out)

        return out

    @staticmethod
    def get_optimizer(optim_name: str, model, lr: float = 0.0001):

        if optim_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optim_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Optimizer not configured.")

        return optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        print("Labels inside loss fn: ", labels)
        print("Logits inside loss fn: ", logits)

        self.weight = self.weight.to(logits.device)
        loss = self.criterion(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def configure_criterion(
        weight: torch.Tensor,
        kind: str = "bce",
        gamma: float = 0.0
    ):

        if kind == "bce" and gamma == 0.0:
            criterion = torch.nn.BCELoss(weight=weight)
        elif gamma > 0:
            logging.info(f"Using focal loss to train classifier.")
            assert gamma >= 0, "Make sure gamma is configured for focal loss."
            criterion = FocalLoss(
                alpha=weight.squeeze(),
                gamma=gamma
            )
        else:
            raise NotImplementedError("Loss criterion not implemented.")

        return criterion

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
