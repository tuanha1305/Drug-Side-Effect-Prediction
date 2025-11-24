"""
Loss functions for drug side effect prediction
Strictly following HSTrans paper (Section 3.5)
"""

import torch
import torch.nn as nn
from typing import Optional


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss

    Standard loss for regression tasks as specified in HSTrans paper.
    Paper Equation (16): Sum of squared differences between true and predicted frequencies.

    Note: While Eq (16) uses Sum, 'mean' reduction is standard for
    batch-size independent optimization.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean' (default) or 'sum'
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
        # Ensure shapes match
        if predictions.shape != targets.shape:
            targets = targets.view_as(predictions)

        # Calculate squared error: (y - y_hat)^2
        loss = (predictions - targets) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_name: str = 'mse', **kwargs) -> nn.Module:
    """
    Factory function to get loss function.

    Args:
        loss_name: Must be 'mse' as per HSTrans paper.
        **kwargs: Arguments for MSELoss (e.g., reduction)

    Returns:
        loss_fn: Loss function module
    """
    loss_name = loss_name.lower()

    if loss_name == 'mse':
        return MSELoss(**kwargs)

    # Paper explicitly states MSE is used. Other losses are removed.
    raise ValueError(
        f"HSTrans paper only uses 'mse' loss (Section 3.5). "
        f"Got: {loss_name}"
    )


if __name__ == "__main__":
    # Test cleaned loss function
    print("=" * 60)
    print("Testing HSTrans Loss Function (MSE)")
    print("=" * 60)

    # Create dummy data
    batch_size = 5
    predictions = torch.tensor([1.5, 2.0, 3.5, 4.0, 0.5])
    targets = torch.tensor([1.0, 2.0, 4.0, 5.0, 0.0])

    print(f"\nPredictions: {predictions}")
    print(f"Targets:     {targets}")

    # Test MSE
    loss_fn = get_loss_function('mse', reduction='mean')
    loss = loss_fn(predictions, targets)

    # Manual check: ((0.5^2 + 0 + 0.5^2 + 1.0^2 + 0.5^2) / 5) = (0.25 + 0 + 0.25 + 1 + 0.25) / 5 = 1.75 / 5 = 0.35
    print(f"\nMSE Loss (mean): {loss.item():.4f}")
    assert abs(loss.item() - 0.35) < 1e-6

    print("✓ Loss calculation is correct according to HSTrans paper.")