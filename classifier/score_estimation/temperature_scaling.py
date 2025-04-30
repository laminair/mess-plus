import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import log_loss


class TemperatureScaling:
    """
    A class for performing temperature scaling for multi-label classification model calibration.
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the temperature scaling model."""
        self.temperature = nn.Parameter(torch.ones(1).to(device))
        self.device = device

    def fit(self, logits, true_labels, max_iter=100, lr=0.01):
        """
        Learn the optimal temperature value using validation data.

        Args:
            logits: Raw logits from the model (before sigmoid)
            true_labels: True binary labels
            max_iter: Maximum number of optimization iterations
            lr: Learning rate for optimization

        Returns:
            Optimal temperature value
        """
        logits = torch.FloatTensor(logits).to(self.device)
        true_labels = torch.FloatTensor(true_labels).to(self.device)

        # Define the temperature parameter and optimizer
        temperature = nn.Parameter(torch.ones(1).to(self.device))
        optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        # Define the loss function (binary cross-entropy for multi-label)
        criterion = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            # Apply temperature scaling
            scaled_logits = logits / temperature
            loss = criterion(scaled_logits, true_labels)
            loss.backward()
            return loss

        # Optimize the temperature parameter
        optimizer.step(eval_loss)

        # Save the optimal temperature
        self.temperature.data = temperature.data

        return self.temperature.item()

    def calibrate(self, logits):
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw logits from the model (before sigmoid)

        Returns:
            Calibrated logits
        """
        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits).to(self.device)

        with torch.no_grad():
            calibrated_logits = logits / self.temperature

        if isinstance(calibrated_logits, torch.Tensor):
            return calibrated_logits.cpu().numpy()
        return calibrated_logits

    def evaluate(self, logits, calibrated_logits, true_labels):
        """
        Evaluate calibration performance by comparing log loss.

        Args:
            logits: Raw logits (before sigmoid)
            calibrated_logits: Temperature-scaled logits
            true_labels: True binary labels

        Returns:
            Dictionary with uncalibrated and calibrated log loss
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        if isinstance(calibrated_logits, torch.Tensor):
            calibrated_logits = calibrated_logits.cpu().numpy()
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().numpy()

        # Calculate probabilities
        uncalibrated_probs = 1 / (1 + np.exp(-logits))
        calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))

        # Calculate log loss for each class and average
        uncalibrated_loss = log_loss(true_labels, uncalibrated_probs)
        calibrated_loss = log_loss(true_labels, calibrated_probs)

        return {
            "uncalibrated_loss": uncalibrated_loss,
            "calibrated_loss": calibrated_loss,
            "improvement": uncalibrated_loss - calibrated_loss
        }


# Updated predict method with temperature scaling
def predict_with_calibration(self, texts, temperature_scaler=None):
    """
    Predict labels for the given texts with optional temperature scaling.

    Args:
        texts: Input texts to classify
        temperature_scaler: Trained TemperatureScaling instance (optional)

    Returns:
        Tuple of (binary predictions, calibrated probabilities)
    """
    self.model.eval()

    # Tokenize texts
    inputs = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=self.max_length,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    # Make predictions
    with torch.no_grad():
        outputs = self.model.forward(**inputs)
        logits = outputs["logits"]

        # Apply temperature scaling if provided
        if temperature_scaler is not None:
            # Convert to numpy for calibration if needed
            if isinstance(logits, torch.Tensor):
                logits_np = logits.cpu().numpy()
            else:
                logits_np = logits

            # Apply calibration
            calibrated_logits = temperature_scaler.calibrate(logits_np)

            # Convert back to tensor if needed
            if isinstance(logits, torch.Tensor):
                calibrated_logits = torch.FloatTensor(calibrated_logits).to(self.device)

            # Calculate probabilities with calibrated logits
            probs = torch.sigmoid(torch.tensor(calibrated_logits)).cpu().numpy()
        else:
            # Use uncalibrated logits
            probs = torch.sigmoid(logits).cpu().numpy()

        predictions = (probs >= self.threshold).astype(int)

    return predictions, probs
