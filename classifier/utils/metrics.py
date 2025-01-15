import torch

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from typing import Tuple, Dict, Any


def compute_scores(probs, labels, stage: str, include_confusion_matrix: bool = True) -> Tuple[Dict, Any]:
    preds = torch.where(probs > 0.5, 1.0, 0.0)
    preds = preds.double()

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    acc = accuracy_score(labels, preds, normalize=True)
    f1 = f1_score(labels, preds, average="weighted")
    prec = precision_score(labels, preds, average="weighted")
    reca = recall_score(labels, preds, average="weighted")

    confusion_mx = None
    if include_confusion_matrix:
        confusion_mx = confusion_matrix(labels, preds)

    return {f"{stage}/accuracy": acc, f"{stage}/f1": f1, f"{stage}/precision": prec, f"{stage}/recall": reca}, confusion_mx