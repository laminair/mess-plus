import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch import nn


class WeightedLossTrainer(pl.Trainer):

    def __init__(self, gamma=2.0, alpha=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = gamma
        self.focal_alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        self.weight = self.weight.to(logits.device)
        focal_loss_fn = FocalLoss(gamma=self.focal_gamma, alpha=self.focal_alpha)
        loss = focal_loss_fn(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

    def update_class_weights(self, df) -> None:
        """
        When doing online learning we can only update the class weights "as we go".
        :param df: Pandas dataframe with 'label' column containing preference labels the classifier has seen up to
                    timestep t
        :return:
        """
        alpha_vec = self.compute_class_weights(df)  
        self.focal_alpha = alpha_vec.squeeze()

    @staticmethod
    def compute_class_weights(df) -> torch.Tensor:

        if type(df["label"].iloc[0]) is str:
            df["label"] = df["label"].apply(lambda x: eval(x))

        label_matrix = np.stack(df["label"].values, axis=0)
        num_samples, num_labels = label_matrix.shape

        counts = label_matrix.sum(axis=0)  
        freq = counts / (num_samples + 1e-9) 

        weights = 1.0 / (freq + 1e-9)
        weights = weights / weights.mean()  
        weights = weights.reshape(1, num_labels)

        return torch.as_tensor(weights, dtype=torch.float)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        :param gamma: Focusing parameter.
        :param alpha: Weight for positive examples. Could be:
            - None (equal weight to each class)
            - A float in [0,1] if single alpha for all classes
            - A tensor of shape [num_labels], if different for each label dimension
        :param reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        :param logits: shape [batch_size, num_labels], raw outputs (if you want
                       internal Sigmoid) or probabilities in [0,1].
        :param targets: shape [batch_size, num_labels], 0/1 multi-label targets.
        :return: focal loss (scalar)
        """
        probs = torch.sigmoid(logits)

        eps = 1e-8
        p = probs.clamp(eps, 1.0 - eps)  # avoid log(0)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)  # shape [num_labels] or single float
        else:
            alpha_t = 1.0

        pt = p * targets + (1 - p) * (1 - targets)
        focal_factor = (1 - pt) ** self.gamma
        
        # Weighted BCE:
        # -alpha * targets * log(p)
        loss_pos = -alpha_t * targets * focal_factor * torch.log(p)

        if isinstance(alpha_t, torch.Tensor) or isinstance(alpha_t, float):
            alpha_neg = 1.0 - alpha_t if isinstance(alpha_t, float) else (1.0 - alpha_t)
        else:
            alpha_neg = 1.0
        loss_neg = -alpha_neg * (1 - targets) * focal_factor * torch.log(1 - p)
        
        loss_out = loss_pos + loss_neg  # shape: [batch_size, num_labels]

        if self.reduction == 'none':
            return loss_out
        elif self.reduction == 'sum':
            return loss_out.sum()
        else:  
            return loss_out.mean()