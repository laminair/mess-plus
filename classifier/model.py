import pandas as pd
import torch
import numpy as np
import os
import time
import logging
import wandb

from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModel, AutoConfig, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from zeus.monitor import ZeusMonitor

from typing import List


# Get absolute path of current file
FILE_FOLDER_PATH = Path(__file__).resolve().parent
NUM_GPUS = torch.cuda.device_count()


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


torch.set_float32_matmul_precision('high')


class BERTClassifier(nn.Module):
    """
    Custom BERT model with a classification layer for multi-label text classification.
    """

    def __init__(
            self,
            model_name="answerdotai/ModernBERT-base",
            num_labels=None,
            dropout_rate=0.2,
            freeze_bert_layers=False
    ):
        super(BERTClassifier, self).__init__()

        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        # Freeze BERT layers if specified
        if freeze_bert_layers:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Get the hidden size from the BERT configuration
        hidden_size = self.config.hidden_size

        # Classification layers
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)

        Returns:
            dict: Dictionary containing logits
        """
        # Pass input through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
        )

        # Get the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification layer
        logits = self.classifier(pooled_output)

        return {"logits": logits}


class MultilabelBERTClassifier:

    def __init__(
            self,
            model_name="answerdotai/ModernBERT-base",
            num_labels=None,
            learning_rate=2e-5,
            momentum=0.0,
            weight_decay=0.01,
            batch_size=16,
            max_length=128,
            warmup_ratio=0.1,
            threshold=0.5,
            dropout_rate=0.1,
            freeze_bert_layers=False,
            device=None,
            checkpoint_path=None,
            disable_tqdm=False,
            **kwargs
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_length = max_length
        self.warmup_ratio = warmup_ratio
        self.threshold = threshold
        self.dropout_rate = dropout_rate
        self.freeze_bert_layers = freeze_bert_layers

        # Determine device (GPU/CPU)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.energy_monitor = ZeusMonitor(gpu_indices=[i for i in range(NUM_GPUS)], approx_instant_energy=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Config for MESS+
        # self.config = config

        self.disable_tqdm = disable_tqdm
        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path.startswith("/"):
            checkpoint_pth = f"{FILE_FOLDER_PATH}/{self.checkpoint_path}"
            self.checkpoint_path = checkpoint_pth

        self.train_ctr = 0

    def fit(self, train_dataset, val_dataset = None, epochs=5, early_stopping_patience=3, ctr: int = None, online_learn: bool = False):
        """Train the model with early stopping based on validation performance."""
        # Load the model if not already done
        self.make_model_if_not_exists(train_dataset=train_dataset)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        if online_learn is False:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )

        # Loss function for multi-label classification
        criterion = nn.BCEWithLogitsLoss()

        # Training loop with early stopping
        best_val_f1 = 0
        patience_counter = 0

        metrics_dict = {
            "classifier/train_step_energy": 0.0
        }
        for epoch in range(epochs):
            start_time = time.time()

            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0

            # Create a tqdm progress bar with statistics
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs} [Training]",
                leave=True,
                position=0,
                disable=self.disable_tqdm
            )

            # Initialize running statistics
            running_loss = 0.0
            for idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Clear gradients
                self.optimizer.zero_grad()

                # Forward pass
                self.energy_monitor.begin_window("training_step")
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    # token_type_ids=batch.get('token_type_ids', None)
                )
                measurement = self.energy_monitor.end_window("training_step")
                metrics_dict["classifier/train_step_energy"] += sum([val for val in measurement.gpu_energy.values()])

                logits = outputs["logits"]
                loss = criterion(logits, batch['labels'])

                # # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                self.optimizer.step()
                # scheduler.step()

                # Update statistics
                batch_loss = loss.item()
                train_loss += batch_loss
                train_steps += 1

                # Update running loss (for the progress bar)
                running_loss = 0.9 * running_loss + 0.1 * batch_loss if train_steps > 1 else batch_loss

                # Update progress bar with current loss
                progress_bar.set_postfix({
                    'loss': f"{running_loss:.4f}",
                    'batch_loss': f"{batch_loss:.4f}",
                    'lr': f"{self.learning_rate}"  # f"{scheduler.get_last_lr()[0]:.1e}"
                })

                self.train_ctr += 1

            avg_train_loss = train_loss / train_steps

            metrics_dict["classifier/step_train_loss"] = avg_train_loss

            if online_learn is False:
                # Validation
                self.model.eval()
                val_loss = 0
                val_steps = 0
                all_preds = []
                all_labels = []

                # Create a validation progress bar
                val_progress = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{epochs} [Validation]",
                    leave=True,
                    position=0,
                    disable=self.disable_tqdm

                )

                with torch.no_grad():
                    for batch in val_progress:
                        # Move batch to device
                        batch = {k: v.to(self.device) for k, v in batch.items()}

                        # Forward pass
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            # token_type_ids=batch.get('token_type_ids', None)
                        )

                        logits = outputs["logits"]
                        loss = criterion(logits, batch['labels'])
                        batch_loss = loss.item()
                        val_loss += batch_loss
                        val_steps += 1

                        # Update validation progress bar
                        val_progress.set_postfix({
                            'val_loss': f"{batch_loss:.4f}",
                            'avg_val_loss': f"{val_loss / val_steps:.4f}"
                        })

                        # Convert logits to predictions
                        preds = torch.sigmoid(logits).cpu().numpy()
                        labels = batch['labels'].cpu().numpy()

                        # Apply threshold
                        preds = (preds >= self.threshold).astype(int)

                        all_preds.append(preds)
                        all_labels.append(labels)

                # Combine predictions and calculate metrics
                all_preds = np.vstack(all_preds)
                all_labels = np.vstack(all_labels)

                avg_val_loss = val_loss / val_steps
                val_accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
                val_precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
                val_recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
                val_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
                val_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

                # Print epoch summary with more detailed metrics
                epoch_time = time.time() - start_time
                logger.info(f"Epoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s")
                logger.info(f"  Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
                logger.info(
                    f"  Val Metrics - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, F1(macro): {val_f1_macro:.4f}")
                # logger.info(f"  Val Metrics - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

                # Log epoch metrics
                metrics_dict.update({
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "val/accuracy": val_accuracy,
                    "val/precision_micro": val_precision,
                    "val/recall_micro": val_recall,
                    "val/f1_micro": val_f1,
                    "val/f1_macro": val_f1_macro,
                    # "val/f1_weighted": val_f1_weighted,
                    "time/epoch_seconds": epoch_time
                })

                # Print per-label metrics if there are fewer than 10 labels
                if self.num_labels < 10:
                    per_label_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
                    per_label_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
                    per_label_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

                    logger.info("  Per-label metrics:")
                    for i in range(self.num_labels):
                        metrics_dict.update({
                            f"val/{i}_f1score": per_label_f1[i],
                            f"val/{i}_recall": per_label_precision[i],
                            f"val/{i}_precision": per_label_recall[i]
                        })

                        logger.info(
                            f"    Label {i}: F1={per_label_f1[i]:.4f}, "
                            f"Prec={per_label_precision[i]:.4f}, "
                            f"Rec={per_label_recall[i]:.4f}"
                        )

                if wandb.run is not None:
                    # We only log to W&B if initialized
                    if ctr is not None:
                        wandb.log(metrics_dict, step=ctr)
                    else:
                        wandb.log(metrics_dict)

                # Early stopping check
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    # Save the best model
                    if online_learn is False:
                        self.get_or_create_path(self.checkpoint_path)
                        self.save_model(f"{self.checkpoint_path}/messplus_classifier.pt")
                        logger.info("Best model saved")
                else:
                    patience_counter += 1
                    logger.warning(f"No improvement: {patience_counter}/{early_stopping_patience}")
                    if patience_counter >= early_stopping_patience:
                        logger.warning("Early stopping triggered!")
                        break

        return metrics_dict

    def evaluate(self, data_loader):
        """Evaluate the model on the given data loader."""
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", disable=self.disable_tqdm):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    # token_type_ids=batch.get('token_type_ids', None)
                )

                logits = outputs["logits"]
                loss = criterion(logits, batch['labels'])
                total_loss += loss.item()

                # Convert logits to predictions
                preds = torch.sigmoid(logits).cpu().numpy()
                labels = batch['labels'].cpu().numpy()

                # Apply threshold
                preds = (preds >= self.threshold).astype(int)

                all_preds.append(preds)
                all_labels.append(labels)

        # Combine predictions and calculate metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        metrics = {
            'val/loss': total_loss / len(data_loader),
            'val/accuracy': accuracy_score(all_labels.flatten(), all_preds.flatten()),
            'val/precision': precision_score(all_labels, all_preds, average='micro', zero_division=0),
            'val/recall': recall_score(all_labels, all_preds, average='micro', zero_division=0),
            'val/f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
            'val/f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'val/f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }

        return metrics

    def predict(self, texts: List[str]):
        """Predict labels for the given texts."""
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
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions = (probs >= self.threshold).astype(int)


        return predictions, probs

    def make_model_if_not_exists(self, train_dataset=None, num_labels: int = None):
        if not hasattr(self, 'model'):
            if self.num_labels is None:
                # Infer from the dataset
                sample = train_dataset[0]
                if isinstance(sample, dict) and 'labels' in sample and num_labels is None:
                    self.num_labels = sample['labels'].shape[0]
                elif num_labels is not None:
                    self.num_labels = num_labels
                else:
                    raise ValueError("Could not infer num_labels from dataset. Please specify num_labels.")

            logger.info(f"Initializing custom BERTClassifier: {self.model_name} with {self.num_labels} labels")
            self.model = BERTClassifier(
                model_name=self.model_name,
                num_labels=self.num_labels,
                dropout_rate=self.dropout_rate,
                freeze_bert_layers=self.freeze_bert_layers
            )
            self.model.to(self.device)

            # Setup optimizer and scheduler
            # Use param groups to apply different learning rates to BERT and classification layers
            if self.freeze_bert_layers:
                # If BERT is frozen, only train the classifier
                parameters = self.model.classifier.parameters()
            else:
                # Otherwise, use different learning rates
                bert_params = {'params': self.model.bert.parameters(), 'lr': self.learning_rate}
                classifier_params = {'params': self.model.classifier.parameters(), 'lr': self.learning_rate * 10}
                parameters = [bert_params, classifier_params]

            self.optimizer = optim.SGD(
                parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )

    def save_model(self, path):
        """Save the model, tokenizer and configuration."""
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'max_length': self.max_length,
                'threshold': self.threshold
            }
        }, path)

        # Save tokenizer
        self.tokenizer.save_pretrained(f"{path}_tokenizer")

    def load_model(self, path):
        """Load the model from the given path."""
        checkpoint = torch.load(f"{self.checkpoint_path}/messplus_classifier.pt", map_location=self.device)
        # Update configuration
        config = checkpoint['config']
        self.model_name = config['model_name']
        self.num_labels = config['num_labels']
        self.max_length = config['max_length']
        self.threshold = config['threshold']

        # Load the model if not already initialized
        if not hasattr(self, 'model'):
            self.model = BERTClassifier(
                model_name=self.model_name,
                num_labels=self.num_labels,
                dropout_rate=self.dropout_rate,
                freeze_bert_layers=self.freeze_bert_layers
            )

        # Convert the state dict keys from "bert.*" to "model.*"
        state_dict = checkpoint['model_state_dict']

        # Load state dict
        self.model.cpu()
        # We still use strict=False since there might be some keys that don't match exactly
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)

        # Load tokenizer if saved
        if hasattr(self, 'tokenizer') and os.path.exists(f"{self.checkpoint_path}/messplus_classifier.pt_tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(f"{self.checkpoint_path}/messplus_classifier.pt_tokenizer")

        return self

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for the DataLoader."""
        if isinstance(batch[0], dict):
            # If the dataset returns dictionaries
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])

            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

            # Add token_type_ids if present
            if 'token_type_ids' in batch[0]:
                token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
                result['token_type_ids'] = token_type_ids

            return result
        else:
            # If the dataset returns tuples/lists
            raise ValueError("Expected dataset to return dictionaries")

    @staticmethod
    def get_or_create_path(path):
        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory {path}")
        else:
            logger.info(f"Directory {path} existis. Reusing...")

        return path


class ContinualMultilabelBERTClassifier(MultilabelBERTClassifier):
    """
    Extension of MultilabelBERTClassifier that supports incremental training
    with new data that arrives over time.
    """

    def __init__(self, *args, **kwargs):
        # Initialize with memory buffer size if provided, otherwise None
        self.memory_size = kwargs.pop('memory_size', None)
        self.memory_buffer = []

        # Pass remaining arguments to parent constructor
        super().__init__(*args, **kwargs)

        # Track training history across all increments
        self.training_history = {
            "train/avg_loss": [],
            "train/avg_accuracy": [],
            "train/step_accuracy": [],
            "val/loss": [],
            "val/f1": [],
            "val/precision": [],
            "val/recall": []
        }

        # Track data seen so far
        self.data_seen = 0

    def update_memory_buffer(self, dataset, strategy='random'):
        """
        Update memory buffer with samples from new dataset.

        Args:
            dataset: The new dataset to sample from
            strategy: Strategy for memory update ('random', 'recent', 'balanced')
        """
        if self.memory_size is None:
            return

        if strategy == 'random':
            # Randomly sample from new dataset to add to memory
            num_to_add = min(len(dataset), self.memory_size - len(self.memory_buffer))
            if num_to_add > 0:
                indices = torch.randperm(len(dataset))[:num_to_add]
                for idx in indices:
                    self.memory_buffer.append(dataset[idx])

        elif strategy == 'recent':
            # Keep most recent examples
            new_buffer = []
            indices = torch.randperm(len(dataset))[:self.memory_size]
            for idx in indices:
                new_buffer.append(dataset[idx])
            self.memory_buffer = new_buffer

        elif strategy == 'balanced':
            # Try to keep a balanced sample of all classes seen so far
            # This is more complex and depends on your specific use case
            # Simplified implementation here
            self.memory_buffer = []
            indices = torch.randperm(len(dataset))[:self.memory_size]
            for idx in indices:
                self.memory_buffer.append(dataset[idx])

        # If memory buffer is too large, trim it
        if len(self.memory_buffer) > self.memory_size:
            self.memory_buffer = self.memory_buffer[-self.memory_size:]

    def create_memory_dataset(self):
        """Create a dataset from the memory buffer."""
        if not self.memory_buffer:
            return None

        # Create a dataset from memory buffer items
        from torch.utils.data import Dataset

        class MemoryDataset(Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

        return MemoryDataset(self.memory_buffer)

    def incremental_fit(
            self,
            new_train_dataset,
            new_val_dataset: pd.DataFrame = None,
            epochs: int = 1,
            memory_strategy: str = "random",
            learning_rate: float = None,
            reset_optimizer: bool = False,
            regularization_lambda: float = 0.0,
            timestamp: int = 0,
    ):
        """
        Incrementally train the model on new data.

        Args:
            new_train_dataset: New training dataset
            new_val_dataset: New validation dataset
            epochs: Number of epochs to train for
            memory_strategy: Strategy for memory buffer updates
            learning_rate: Optional learning rate override
            reset_optimizer: Whether to reset the optimizer
            regularization_lambda: Strength of regularization to previous model
            timestamp: Timestamp of request
        """
        # Load the model if not already done
        self.make_model_if_not_exists(train_dataset=new_train_dataset)

        # Store old model for regularization if needed
        old_model = None
        if regularization_lambda > 0:
            old_model = type(self.model).from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="multi_label_classification"
            )
            old_model.load_state_dict(self.model.state_dict())
            old_model.to(self.device)
            old_model.eval()

        # Create combined dataset (new data + memory)
        memory_dataset = self.create_memory_dataset()

        if memory_dataset:
            from torch.utils.data import ConcatDataset
            combined_train = ConcatDataset([new_train_dataset, memory_dataset])
            logger.info(
                f"Training on combined dataset: {len(new_train_dataset)} new + {len(memory_dataset)} memory = {len(combined_train)} total")
        else:
            combined_train = new_train_dataset
            logger.info(f"Training on new data only: {len(new_train_dataset)} examples")

        # Create data loaders
        train_loader = DataLoader(
            combined_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        if new_val_dataset is not None:

            val_loader = DataLoader(
                new_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )

        # Setup optimizer and scheduler
        if learning_rate:
            self.learning_rate = learning_rate

        if reset_optimizer or not hasattr(self, 'optimizer'):
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Loss function for multi-label classification
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        val_metrics = {}
        best_val_f1 = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0

            all_labels = []
            all_preds = []
            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs} [Training]",
                disable=self.disable_tqdm
            ):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Clear gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    # token_type_ids=batch.get('token_type_ids', None)
                )

                logits = outputs["logits"]
                loss = criterion(logits, batch['labels'])

                # Add regularization loss to prevent catastrophic forgetting
                if old_model is not None:
                    with torch.no_grad():
                        old_outputs = old_model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            # token_type_ids=batch.get('token_type_ids', None)
                        )

                    # L2 regularization to keep predictions similar to the old model
                    reg_loss = torch.nn.functional.mse_loss(logits, old_outputs["logits"])
                    loss += regularization_lambda * reg_loss

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_steps += 1

                # Convert logits to predictions
                preds = torch.sigmoid(logits).detach().cpu().numpy()
                labels = batch['labels'].cpu().numpy()

                # Apply threshold
                preds = (preds >= self.threshold).astype(int)

                all_preds.append(preds)
                all_labels.append(labels)

                # Combine predictions and calculate metrics
                all_preds = np.vstack(all_preds)
                all_labels = np.vstack(all_labels)

                acc_score = accuracy_score(all_labels.flatten(), all_preds.flatten())
                self.training_history["train/avg_accuracy"].append(acc_score)

                if wandb.run is not None:
                    wandb.log({
                        "train/step_loss": loss.item(),
                        "train/step_accuracy": acc_score,
                        "train/avg_accuracy": sum(self.training_history["train/avg_accuracy"]) / timestamp if timestamp > 0 else 0
                    }, step=timestamp)

            avg_train_loss = train_loss / train_steps

            # Validation
            if new_val_dataset is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['val/loss']
                val_f1 = val_metrics['val/f1_micro']

            # Print epoch summary
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s")

            if new_val_dataset is not None:
                logger.info(f"  Train Loss: {avg_train_loss:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
                logger.info(f"  Precision: {val_metrics['val/precision']:.4f}, Recall: {val_metrics['val/recall']:.4f}")

            # Update training history
            self.training_history['train/avg_loss'].append(avg_train_loss)

            if new_val_dataset is not None:
                self.training_history['val/loss'].append(val_loss)
                self.training_history['val/f1'].append(val_f1)
                self.training_history['val/precision'].append(val_metrics['val/precision'])
                self.training_history['val/recall'].append(val_metrics['val/recall'])

            # Save model if it's the best so far
            if new_val_dataset is not None and val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # Save the best model
                # self.save_model(f"{FILE_FOLDER_PATH}/checkpoints/best_model_increment_{self.data_seen}.pt")
                # logger.info("  Best model saved!")

        # Update memory buffer with samples from new data
        self.update_memory_buffer(new_train_dataset, strategy=memory_strategy)

        # Update data seen counter
        self.data_seen += len(new_train_dataset)

        if new_val_dataset is not None and wandb.run is not None:
            wandb.log(val_metrics, step=timestamp)

        if new_val_dataset is not None:
            return val_metrics
        else:
            return None

    def save_checkpoint(self, path):
        """
        Save a complete checkpoint including model, optimizer, scheduler, and memory buffer.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'memory_buffer': self.memory_buffer,
            'data_seen': self.data_seen,
            'training_history': self.training_history,
            'config': {
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'batch_size': self.batch_size,
                'max_length': self.max_length,
                'warmup_ratio': self.warmup_ratio,
                'threshold': self.threshold,
                'memory_size': self.memory_size
            }
        }, path)

        # Save tokenizer
        self.tokenizer.save_pretrained(f"{path}_tokenizer")

    def load_checkpoint(self, path):
        """
        Load a complete checkpoint including model, optimizer, scheduler, and memory buffer.
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Update configuration
        config = checkpoint['config']
        self.model_name = config['model_name']
        self.num_labels = config['num_labels']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.batch_size = config['batch_size']
        self.max_length = config['max_length']
        self.warmup_ratio = config['warmup_ratio']
        self.threshold = config['threshold']
        self.memory_size = config['memory_size']

        # Load the model if not already initialized
        if not hasattr(self, 'model'):
            self.model = AutoModel.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                problem_type="multi_label_classification"
            )

        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Load optimizer if available
        if checkpoint['optimizer_state_dict']:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load memory buffer
        self.memory_buffer = checkpoint['memory_buffer']

        # Load training history and data seen
        self.training_history = checkpoint['training_history']
        self.data_seen = checkpoint['data_seen']

        # Load tokenizer if saved
        if os.path.exists(f"{path}_tokenizer"):
            self.tokenizer = AutoTokenizer.from_pretrained(f"{path}_tokenizer")

        return self
