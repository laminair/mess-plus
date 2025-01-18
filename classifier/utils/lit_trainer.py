import numpy as np
import pytorch_lightning as pl
import torch

from torch import nn


class WeightedLossTrainer(pl.Trainer):

    def __init__(self, label_list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.ones(size=[1, 3])

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        self.weight = self.weight.to(logits.device)
        loss_fct = nn.BCELoss(weight=self.weight)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def update_class_weights(self, label_list) -> None:
        """
        When doing online learning we can only update the class weights "as we go".
        :param label_list: List of preference labels the classifier has seen up to timestep t
        :return:
        """
        self.weight = self.compute_class_weights(label_list)

    @staticmethod
    def compute_class_weights(labels) -> torch.Tensor:
        unique_labels = set(labels)
        num_labels = len(unique_labels)

        counts = np.bincount(labels, minlength=num_labels)
        weights = 1.0 / (counts + 1e-9)
        weights = weights / weights.mean()

        return torch.as_tensor(weights, dtype=torch.float)
