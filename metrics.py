"""
Metrics for drug side effect prediction
Comprehensive metrics for regression and classification evaluation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from scipy.stats import pearsonr, spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Regression Metrics
# ============================================================================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        mse: Mean squared error
    """
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        rmse: Root mean squared error
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        mae: Mean absolute error
    """
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (coefficient of determination)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        r2: R-squared score
    """
    return float(r2_score(y_true, y_pred))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Pearson correlation coefficient

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        correlation: Pearson correlation coefficient
        p_value: Two-tailed p-value
    """
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0, 1.0

    try:
        corr, p_val = pearsonr(y_true, y_pred)
        return float(corr), float(p_val)
    except:
        return 0.0, 1.0


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Spearman correlation coefficient

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        correlation: Spearman correlation coefficient
        p_value: Two-tailed p-value
    """
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0, 1.0

    try:
        corr, p_val = spearmanr(y_true, y_pred)
        return float(corr), float(p_val)
    except:
        return 0.0, 1.0


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error

    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero

    Returns:
        mape: Mean absolute percentage error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = np.abs(y_true) > epsilon

    if mask.sum() == 0:
        return 0.0

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ============================================================================
# Classification Metrics
# ============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Accuracy score

    Args:
        y_true: True labels
        y_pred: Predicted probabilities or labels
        threshold: Threshold for binary classification

    Returns:
        accuracy: Accuracy score
    """
    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    return float(accuracy_score(y_true_binary, y_pred_binary))


def precision(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
        average: str = 'binary'
) -> float:
    """
    Precision score

    Args:
        y_true: True labels
        y_pred: Predicted probabilities or labels
        threshold: Threshold for binary classification
        average: 'binary', 'micro', 'macro', 'weighted'

    Returns:
        precision: Precision score
    """
    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    return float(precision_score(y_true_binary, y_pred_binary, average=average, zero_division=0))


def recall(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
        average: str = 'binary'
) -> float:
    """
    Recall score (Sensitivity, True Positive Rate)

    Args:
        y_true: True labels
        y_pred: Predicted probabilities or labels
        threshold: Threshold for binary classification
        average: 'binary', 'micro', 'macro', 'weighted'

    Returns:
        recall: Recall score
    """
    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    return float(recall_score(y_true_binary, y_pred_binary, average=average, zero_division=0))


def f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
        average: str = 'binary'
) -> float:
    """
    F1 score (harmonic mean of precision and recall)

    Args:
        y_true: True labels
        y_pred: Predicted probabilities or labels
        threshold: Threshold for binary classification
        average: 'binary', 'micro', 'macro', 'weighted'

    Returns:
        f1: F1 score
    """
    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    return float(f1_score(y_true_binary, y_pred_binary, average=average, zero_division=0))


def specificity(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Specificity score (True Negative Rate)

    Args:
        y_true: True labels
        y_pred: Predicted probabilities or labels
        threshold: Threshold for binary classification

    Returns:
        specificity: Specificity score
    """
    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Area Under the ROC Curve

    Args:
        y_true: True labels
        y_pred: Predicted probabilities

    Returns:
        auc: AUC-ROC score
    """
    y_true_binary = (y_true != 0).astype(int)

    if len(np.unique(y_true_binary)) < 2:
        return 0.0

    try:
        return float(roc_auc_score(y_true_binary, y_pred))
    except:
        return 0.0


def auc_pr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Area Under the Precision-Recall Curve (Average Precision)

    Args:
        y_true: True labels
        y_pred: Predicted probabilities

    Returns:
        auc_pr: AUC-PR score (Average Precision)
    """
    y_true_binary = (y_true != 0).astype(int)

    if len(np.unique(y_true_binary)) < 2:
        return 0.0

    try:
        return float(average_precision_score(y_true_binary, y_pred))
    except:
        return 0.0


def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
) -> Dict[str, int]:
    """
    Get confusion matrix components

    Args:
        y_true: True labels
        y_pred: Predicted probabilities or labels
        threshold: Threshold for binary classification

    Returns:
        cm_dict: Dictionary with TP, TN, FP, FN
    """
    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    return {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Balanced accuracy (average of sensitivity and specificity)

    Args:
        y_true: True labels
        y_pred: Predicted probabilities or labels
        threshold: Threshold for binary classification

    Returns:
        balanced_acc: Balanced accuracy score
    """
    sens = recall(y_true, y_pred, threshold)
    spec = specificity(y_true, y_pred, threshold)

    return (sens + spec) / 2


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Matthews Correlation Coefficient

    Args:
        y_true: True labels
        y_pred: Predicted probabilities or labels
        threshold: Threshold for binary classification

    Returns:
        mcc: Matthews correlation coefficient
    """
    from sklearn.metrics import matthews_corrcoef as mcc_sklearn

    y_true_binary = (y_true != 0).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    return float(mcc_sklearn(y_true_binary, y_pred_binary))


# ============================================================================
# Per-Drug Metrics
# ============================================================================

def per_drug_auc(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        drug_ids: np.ndarray
) -> Tuple[float, List[float]]:
    """
    Calculate AUC per drug

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        drug_ids: Drug identifiers

    Returns:
        mean_auc: Mean AUC across drugs
        drug_aucs: List of AUC for each drug
    """
    unique_drugs = np.unique(drug_ids)
    drug_aucs = []

    for drug_id in unique_drugs:
        mask = drug_ids == drug_id
        drug_y_true = y_true[mask]
        drug_y_pred = y_pred[mask]

        # Convert to binary
        drug_y_true_binary = (drug_y_true != 0).astype(int)

        # Skip if only one class
        if len(np.unique(drug_y_true_binary)) < 2:
            continue

        try:
            auc = roc_auc_score(drug_y_true_binary, drug_y_pred)
            drug_aucs.append(auc)
        except:
            continue

    mean_auc = np.mean(drug_aucs) if len(drug_aucs) > 0 else 0.0

    return float(mean_auc), drug_aucs


def per_drug_aupr(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        drug_ids: np.ndarray
) -> Tuple[float, List[float]]:
    """
    Calculate AUPR per drug

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        drug_ids: Drug identifiers

    Returns:
        mean_aupr: Mean AUPR across drugs
        drug_auprs: List of AUPR for each drug
    """
    unique_drugs = np.unique(drug_ids)
    drug_auprs = []

    for drug_id in unique_drugs:
        mask = drug_ids == drug_id
        drug_y_true = y_true[mask]
        drug_y_pred = y_pred[mask]

        # Convert to binary
        drug_y_true_binary = (drug_y_true != 0).astype(int)

        # Skip if only one class
        if len(np.unique(drug_y_true_binary)) < 2:
            continue

        try:
            aupr = average_precision_score(drug_y_true_binary, drug_y_pred)
            drug_auprs.append(aupr)
        except:
            continue

    mean_aupr = np.mean(drug_auprs) if len(drug_auprs) > 0 else 0.0

    return float(mean_aupr), drug_auprs


# ============================================================================
# Comprehensive Metrics
# ============================================================================

def calculate_all_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all regression metrics

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        metrics: Dictionary of all regression metrics
    """
    # Filter valid samples for correlation
    valid_mask = y_true != 0
    valid_y_true = y_true[valid_mask]
    valid_y_pred = y_pred[valid_mask]

    # Pearson correlation
    pearson_corr, pearson_p = pearson(valid_y_true, valid_y_pred) if len(valid_y_true) > 1 else (0.0, 1.0)

    # Spearman correlation
    spearman_corr, spearman_p = spearman(valid_y_true, valid_y_pred) if len(valid_y_true) > 1 else (0.0, 1.0)

    metrics = {
        'mse': mse(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'r2': r2(y_true, y_pred),
        'pearson': pearson_corr,
        'pearson_pvalue': pearson_p,
        'spearman': spearman_corr,
        'spearman_pvalue': spearman_p,
        'mape': mape(valid_y_true, valid_y_pred) if len(valid_y_true) > 0 else 0.0
    }

    return metrics


def calculate_all_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate all classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        metrics: Dictionary of all classification metrics
    """
    # Confusion matrix
    cm = get_confusion_matrix(y_true, y_pred, threshold)

    metrics = {
        'accuracy': accuracy(y_true, y_pred, threshold),
        'precision': precision(y_true, y_pred, threshold),
        'recall': recall(y_true, y_pred, threshold),
        'f1': f1(y_true, y_pred, threshold),
        'specificity': specificity(y_true, y_pred, threshold),
        'balanced_accuracy': balanced_accuracy(y_true, y_pred, threshold),
        'auc_roc': auc_roc(y_true, y_pred),
        'auc_pr': auc_pr(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred, threshold),
        **cm
    }

    return metrics


def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
        drug_ids: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate all metrics (regression + classification)

    Args:
        y_true: True values/labels
        y_pred: Predicted values/probabilities
        threshold: Classification threshold
        drug_ids: Drug identifiers (optional, for per-drug metrics)

    Returns:
        metrics: Dictionary of all metrics
    """
    # Regression metrics
    reg_metrics = calculate_all_regression_metrics(y_true, y_pred)

    # Classification metrics
    cls_metrics = calculate_all_classification_metrics(y_true, y_pred, threshold)

    # Combine
    all_metrics = {**reg_metrics, **cls_metrics}

    # Per-drug metrics
    if drug_ids is not None:
        drug_auc, _ = per_drug_auc(y_true, y_pred, drug_ids)
        drug_aupr, _ = per_drug_aupr(y_true, y_pred, drug_ids)

        all_metrics['drug_auc'] = drug_auc
        all_metrics['drug_aupr'] = drug_aupr

    return all_metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics

    Args:
        metrics: Dictionary of metrics
        title: Title for the print
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    # Group metrics
    regression_keys = ['mse', 'rmse', 'mae', 'r2', 'pearson', 'spearman', 'mape']
    classification_keys = ['accuracy', 'precision', 'recall', 'f1', 'specificity',
                           'balanced_accuracy', 'auc_roc', 'auc_pr', 'mcc']
    cm_keys = ['TP', 'TN', 'FP', 'FN']
    drug_keys = ['drug_auc', 'drug_aupr']

    # Print regression metrics
    print("\nRegression Metrics:")
    for key in regression_keys:
        if key in metrics:
            print(f"  {key:20s}: {metrics[key]:.4f}")

    # Print classification metrics
    print("\nClassification Metrics:")
    for key in classification_keys:
        if key in metrics:
            print(f"  {key:20s}: {metrics[key]:.4f}")

    # Print confusion matrix
    if all(k in metrics for k in cm_keys):
        print("\nConfusion Matrix:")
        print(f"  TP: {metrics['TP']:5d}  |  FP: {metrics['FP']:5d}")
        print(f"  FN: {metrics['FN']:5d}  |  TN: {metrics['TN']:5d}")

    # Print per-drug metrics
    if any(k in metrics for k in drug_keys):
        print("\nPer-Drug Metrics:")
        for key in drug_keys:
            if key in metrics:
                print(f"  {key:20s}: {metrics[key]:.4f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test metrics
    print("=" * 60)
    print("Testing Metrics")
    print("=" * 60)

    # Create dummy data
    np.random.seed(42)
    n_samples = 100

    y_true = np.random.rand(n_samples)
    y_pred = y_true + np.random.randn(n_samples) * 0.1

    print(f"\nSample size: {n_samples}")
    print(f"y_true range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"y_pred range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

    # Calculate all metrics
    metrics = calculate_all_metrics(y_true, y_pred, threshold=0.5)

    # Print metrics
    print_metrics(metrics, title="All Metrics")

    # Test per-drug metrics
    print("\nTesting per-drug metrics...")
    drug_ids = np.random.randint(0, 10, n_samples)

    drug_auc, drug_aucs = per_drug_auc(y_true, y_pred, drug_ids)
    drug_aupr, drug_auprs = per_drug_aupr(y_true, y_pred, drug_ids)

    print(f"Drug AUC: {drug_auc:.4f} (n={len(drug_aucs)} drugs)")
    print(f"Drug AUPR: {drug_aupr:.4f} (n={len(drug_auprs)} drugs)")

    print("\n" + "=" * 60)
    print("✓ All metrics tests passed!")
    print("=" * 60)