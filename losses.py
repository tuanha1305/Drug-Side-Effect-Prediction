"""
Loss functions for drug side effect prediction
Various loss functions for regression and classification tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss
    Standard loss for regression tasks
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size] or [batch_size, 1]
            targets: [batch_size] or [batch_size, 1]
        """
        loss = (predictions - targets) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error Loss
    More interpretable than MSE
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant for numerical stability
        """
        super(RMSELoss, self).__init__()
        self.eps = eps

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size]
            targets: [batch_size]
        """
        mse = ((predictions - targets) ** 2).mean()
        return torch.sqrt(mse + self.eps)


class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss (L1 Loss)
    More robust to outliers than MSE
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super(MAELoss, self).__init__()
        self.reduction = reduction

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size]
            targets: [batch_size]
        """
        loss = torch.abs(predictions - targets)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss)
    Combination of MSE and MAE - robust to outliers
    """

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            delta: Threshold for switching between MSE and MAE
            reduction: 'mean', 'sum', or 'none'
        """
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size]
            targets: [batch_size]
        """
        diff = torch.abs(predictions - targets)

        # MSE for small errors, MAE for large errors
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss
    For binary classification tasks
    """

    def __init__(self, pos_weight: Optional[float] = None, reduction: str = 'mean'):
        """
        Args:
            pos_weight: Weight for positive class (for imbalanced data)
            reduction: 'mean', 'sum', or 'none'
        """
        super(BCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size] - logits or probabilities
            targets: [batch_size] - 0 or 1
        """
        # Apply sigmoid if predictions are logits
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        # BCE loss
        loss = -(targets * torch.log(predictions + 1e-8) +
                 (1 - targets) * torch.log(1 - predictions + 1e-8))

        # Apply pos_weight if provided
        if self.pos_weight is not None:
            loss = torch.where(
                targets == 1,
                loss * self.pos_weight,
                loss
            )

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss
    Give different weights to different samples
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size]
            targets: [batch_size]
            weights: [batch_size] - sample weights
        """
        loss = weights * (predictions - targets) ** 2

        if self.reduction == 'mean':
            return loss.sum() / weights.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard examples
    """

    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weight for positive class
            gamma: Focusing parameter (larger = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size] - logits or probabilities
            targets: [batch_size] - 0 or 1
        """
        # Apply sigmoid if predictions are logits
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        # BCE loss
        bce_loss = -(targets * torch.log(predictions + 1e-8) +
                     (1 - targets) * torch.log(1 - predictions + 1e-8))

        # Focal weight
        p_t = torch.where(targets == 1, predictions, 1 - predictions)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Final focal loss
        loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined Loss (Regression + Classification)
    Useful for multi-task learning
    """

    def __init__(
            self,
            regression_weight: float = 0.5,
            classification_weight: float = 0.5
    ):
        """
        Args:
            regression_weight: Weight for regression loss
            classification_weight: Weight for classification loss
        """
        super(CombinedLoss, self).__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size]
            targets: [batch_size]
        """
        # Regression loss (continuous values)
        reg_loss = self.mse_loss(predictions, targets)

        # Classification loss (binary: 0 vs non-zero)
        targets_binary = (targets != 0).float()
        cls_loss = self.bce_loss(predictions, targets_binary)

        # Combined loss
        total_loss = (self.regression_weight * reg_loss +
                      self.classification_weight * cls_loss)

        return total_loss


class RankingLoss(nn.Module):
    """
    Ranking Loss (Pairwise)
    Ensure correct relative ordering of predictions
    """

    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            margin: Margin for ranking loss
            reduction: 'mean', 'sum', or 'none'
        """
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size]
            targets: [batch_size]
        """
        # Create pairwise comparisons
        n = predictions.size(0)
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)

        # Sign of target difference
        target_sign = torch.sign(target_diff)

        # Ranking loss: penalize incorrect orderings
        loss = torch.clamp(self.margin - target_sign * pred_diff, min=0)

        # Only consider valid pairs (non-zero target difference)
        mask = (target_diff != 0).float()
        loss = loss * mask

        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name

    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for loss function

    Returns:
        loss_fn: Loss function module
    """
    loss_dict = {
        'mse': MSELoss,
        'rmse': RMSELoss,
        'mae': MAELoss,
        'huber': HuberLoss,
        'bce': BCELoss,
        'focal': FocalLoss,
        'weighted_mse': WeightedMSELoss,
        'combined': CombinedLoss,
        'ranking': RankingLoss
    }

    loss_name = loss_name.lower()

    if loss_name not in loss_dict:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available: {list(loss_dict.keys())}"
        )

    return loss_dict[loss_name](**kwargs)


if __name__ == "__main__":
    # Test loss functions
    print("=" * 60)
    print("Testing Loss Functions")
    print("=" * 60)

    # Create dummy data
    batch_size = 10
    predictions = torch.randn(batch_size)
    targets = torch.randn(batch_size)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    # Test each loss function
    losses = {
        'MSE': MSELoss(),
        'RMSE': RMSELoss(),
        'MAE': MAELoss(),
        'Huber': HuberLoss(delta=1.0),
        'BCE': BCELoss(),
        'Focal': FocalLoss(alpha=0.25, gamma=2.0),
        'Combined': CombinedLoss(),
        'Ranking': RankingLoss(margin=1.0)
    }

    print("\nLoss values:")
    for name, loss_fn in losses.items():
        try:
            if name == 'Weighted MSE':
                weights = torch.ones(batch_size)
                loss = loss_fn(predictions, targets, weights)
            else:
                loss = loss_fn(predictions, targets)
            print(f"  {name:15s}: {loss.item():.4f}")
        except Exception as e:
            print(f"  {name:15s}: Error - {e}")

    # Test factory function
    print("\nTesting factory function:")
    loss_fn = get_loss_function('mse')
    loss = loss_fn(predictions, targets)
    print(f"  MSE (via factory): {loss.item():.4f}")

    # Test with binary targets
    print("\nTesting with binary targets:")
    predictions_binary = torch.sigmoid(torch.randn(batch_size))
    targets_binary = torch.randint(0, 2, (batch_size,)).float()

    bce_loss = BCELoss()
    focal_loss = FocalLoss()

    print(f"  BCE Loss:   {bce_loss(predictions_binary, targets_binary).item():.4f}")
    print(f"  Focal Loss: {focal_loss(predictions_binary, targets_binary).item():.4f}")

    print("\n" + "=" * 60)
    print("✓ All loss function tests passed!")
    print("=" * 60)