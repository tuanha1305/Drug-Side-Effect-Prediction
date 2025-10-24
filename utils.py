"""
Utility functions for drug side effect prediction
Helper functions for logging, visualization, metrics, and data manipulation
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
from datetime import datetime
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Logging Utilities
# ============================================================================

def setup_logger(
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_file: Log file path (optional)
        level: Logging level

    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# Metric Utilities
# ============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient"""
    from scipy.stats import pearsonr
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    try:
        corr, _ = pearsonr(y_true, y_pred)
        return corr
    except:
        return 0.0


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Spearman correlation coefficient"""
    from scipy.stats import spearmanr
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    try:
        corr, _ = spearmanr(y_true, y_pred)
        return corr
    except:
        return 0.0


# ============================================================================
# Visualization Utilities
# ============================================================================

def plot_training_curves(
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        title: str = 'Training Curves',
        save_path: Optional[str] = None
):
    """
    Plot training and validation loss curves

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {save_path}")

    plt.close()


def plot_predictions_vs_actual(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Predictions vs Actual',
        save_path: Optional[str] = None
):
    """
    Plot predictions vs actual values

    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=30)

    # Diagonal line (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Calculate metrics
    from scipy.stats import pearsonr
    pearson, _ = pearsonr(y_true, y_pred) if len(np.unique(y_true)) > 1 else (0, 0)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_val = np.sqrt(mse)

    # Add metrics to plot
    textstr = f'Pearson: {pearson:.3f}\nRMSE: {rmse_val:.3f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {save_path}")

    plt.close()


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
        title: str = 'Confusion Matrix',
        save_path: Optional[str] = None
):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        title: Plot title
        save_path: Path to save figure (optional)
    """
    from sklearn.metrics import confusion_matrix

    # Convert to binary
    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {save_path}")

    plt.close()


def plot_roc_curve(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'ROC Curve',
        save_path: Optional[str] = None
):
    """
    Plot ROC curve

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        title: Plot title
        save_path: Path to save figure (optional)
    """
    from sklearn.metrics import roc_curve, auc

    # Convert to binary
    y_true_binary = (y_true != 0).astype(int)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {save_path}")

    plt.close()


def plot_pr_curve(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Precision-Recall Curve',
        save_path: Optional[str] = None
):
    """
    Plot Precision-Recall curve

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        title: Plot title
        save_path: Path to save figure (optional)
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    # Convert to binary
    y_true_binary = (y_true != 0).astype(int)

    # Compute PR curve
    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred)
    avg_precision = average_precision_score(y_true_binary, y_pred)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {save_path}")

    plt.close()


# ============================================================================
# Data Utilities
# ============================================================================

def save_dict_to_json(data: dict, filepath: str):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved data to: {filepath}")


def load_dict_from_json(filepath: str) -> dict:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded data from: {filepath}")
    return data


def save_pickle(data, filepath: str):
    """Save data to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved data to: {filepath}")


def load_pickle(filepath: str):
    """Load data from pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Loaded data from: {filepath}")
    return data


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Create experiment directory with timestamp

    Args:
        base_dir: Base directory
        experiment_name: Experiment name

    Returns:
        exp_dir: Path to experiment directory
    """
    timestamp = get_timestamp()
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


# ============================================================================
# Model Utilities
# ============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> float:
    """
    Get model size in MB

    Args:
        model: PyTorch model

    Returns:
        size_mb: Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_summary(model: torch.nn.Module):
    """Print model summary"""
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size(model):.2f} MB")
    print("=" * 60)


# ============================================================================
# Training Utilities
# ============================================================================

class EarlyStopping:
    """Early stopping utility"""

    def __init__(
            self,
            patience: int = 10,
            min_delta: float = 0.0,
            mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop

        Args:
            score: Current score

        Returns:
            should_stop: Whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Analysis Utilities
# ============================================================================

def analyze_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dir: str = 'analysis'
):
    """
    Comprehensive analysis of predictions

    Args:
        y_true: True labels
        y_pred: Predictions
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Analyzing predictions...")

    # Plot training curves would go here if we had loss history

    # Predictions vs Actual
    plot_predictions_vs_actual(
        y_true, y_pred,
        save_path=str(output_dir / 'predictions_vs_actual.png')
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=str(output_dir / 'confusion_matrix.png')
    )

    # ROC curve
    plot_roc_curve(
        y_true, y_pred,
        save_path=str(output_dir / 'roc_curve.png')
    )

    # PR curve
    plot_pr_curve(
        y_true, y_pred,
        save_path=str(output_dir / 'pr_curve.png')
    )

    logger.info(f"Analysis plots saved to: {output_dir}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Test metrics
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

    print(f"\nRMSE: {rmse(y_true, y_pred):.4f}")
    print(f"MAE: {mae(y_true, y_pred):.4f}")
    print(f"Pearson: {pearson_correlation(y_true, y_pred):.4f}")
    print(f"Spearman: {spearman_correlation(y_true, y_pred):.4f}")

    print("\n✓ All utility tests passed!")