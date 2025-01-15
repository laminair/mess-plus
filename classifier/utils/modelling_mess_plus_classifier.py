import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from torch import nn

from .metrics import compute_scores


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

    def __init__(self, base_model, model_list: list, hidden_layer_shape: list, optim_name: str, n_classes=10, n_epochs=3, lr=0.0001):
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
        out = self.mean_pooling(out.last_hidden_state, attn_mask)
        out = self.classifier(out)
        out = torch.sigmoid(out)
        return out

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        # We need to make sure we only pool actual tokens and not padding.
        # Please see for reference: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        probs = self(input_ids, attention_mask)
        loss = self.criterion(probs, labels)
        self.log('train/loss', loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": probs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        probs = self(input_ids, attention_mask)
        loss = self.criterion(probs, labels)

        metrics, conf_mx = compute_scores(
            probs,
            labels,
            stage="val",
            include_confusion_matrix=False
        )

        metrics.update({
            "batch_idx": batch_idx,
            "stage": "val"
        })

        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame(metrics, index=[0])])

        self.val_losses.append(loss)
        self.log("val_loss", loss, prog_bar=True, logger=False)
        return loss

    def on_validation_epoch_end(self):

        stage = "val"

        numeric_cols = self.metrics_df.columns.tolist()
        numeric_cols = [i for i in self.metrics_df.columns.tolist() if stage in i]
        epoch_metrics = self.metrics_df.loc[self.metrics_df["stage"] == "val", numeric_cols].mean()
        epoch_metrics = epoch_metrics.to_dict()
        epoch_metrics[f"{stage}/loss"] = np.mean([i.cpu().item() for i in self.val_losses])

        logging.info(epoch_metrics)
        self.log_dict(epoch_metrics)

        self.metrics_df = pd.DataFrame()
        self.val_losses = []

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        if self.optim_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optim_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("Optimizer not configured.")

        return optimizer
