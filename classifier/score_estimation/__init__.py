import numpy as np
import torch

from typing import Union, Optional


class RoutingScoreEstimator:

    def __init__(self):
        self.available_scoring_methods = ["raw"]

    def get_scores(
        self,
        logits: Union[torch.Tensor, np.array],
        labels: Optional[Union[torch.Tensor, np.array]] = None,
        scoring_method: str = "raw"
    ):

        if scoring_method == "raw":
            if len(logits.shape) > 1:
                score = np.cumsum(logits, axis=1) / (np.mean(logits) * logits.shape[1])
            else:
                score = np.cumsum(logits, axis=1) / np.mean(logits)

        else:
            raise NotImplementedError(
                f"Scoring method {scoring_method} not implemented. "
                f"Choose from {self.available_scoring_methods}."
            )

        return score
