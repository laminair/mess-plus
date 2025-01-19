import torch

from torch import nn


class FocalLoss(nn.Module):
	def __init__(self, gamma=2.0, alpha=None, reduction='mean', **kwargs):
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

	def update_alpha(self, squeezed_weight: torch.Tensor):
		self.alpha = squeezed_weight
