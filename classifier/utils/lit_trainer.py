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

    def update_class_weights(self, df) -> None:
        """
        When doing online learning we can only update the class weights "as we go".
        :param df: Pandas dataframe with 'label' column containing preference labels the classifier has seen up to
                    timestep t
        :return:
        """
        self.weight = self.compute_class_weights(df)

    @staticmethod
    def compute_class_weights(df) -> torch.Tensor:

        if type(df["label"].iloc[0]) is str:
            df["label"] = df["label"].apply(lambda x: eval(df["label"]))

        label_matrix = np.stack(df["label"].values, axis=0)
        num_samples, num_labels = label_matrix.shape

        counts = label_matrix.sum(axis=0)  
        freq = counts / (num_samples + 1e-9)

        weights = 1.0 / (freq + 1e-9)
        weights = weights / weights.mean()
        weights = weights.reshape(1, num_labels)

        return torch.as_tensor(weights, dtype=torch.float)
