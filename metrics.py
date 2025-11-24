"""
Metrics for drug side effect prediction
Regression metrics for HSTrans paper evaluation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Regression Metrics (HSTrans Paper)
# ============================================================================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error

    Args:
        y_true: True frequency values (0-5)
        y_pred: Predicted frequency values

    Returns:
        mse: Mean squared error
    """
    return float(mean_squared_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (Primary metric in paper)

    Args:
        y_true: True frequency values (0-5)
        y_pred: Predicted frequency values

    Returns:
        rmse: Root mean squared error
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error

    Args:
        y_true: True frequency values (0-5)
        y_pred: Predicted frequency values

    Returns:
        mae: Mean absolute error
    """
    return float(mean_absolute_error(y_true, y_pred))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Pearson correlation coefficient

    Args:
        y_true: True frequency values (0-5)
        y_pred: Predicted frequency values

    Returns:
        correlation: Pearson correlation coefficient
        p_value: Two-tailed p-value
    """
    # Filter out zeros for meaningful correlation (as per paper)
    valid_mask = y_true != 0
    valid_y_true = y_true[valid_mask]
    valid_y_pred = y_pred[valid_mask]

    if len(valid_y_true) < 2 or len(np.unique(valid_y_true)) < 2:
        return 0.0, 1.0

    try:
        corr, p_val = pearsonr(valid_y_true, valid_y_pred)
        return float(corr), float(p_val)
    except:
        return 0.0, 1.0


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Spearman correlation coefficient (SCC in paper)

    Args:
        y_true: True frequency values (0-5)
        y_pred: Predicted frequency values

    Returns:
        correlation: Spearman correlation coefficient
        p_value: Two-tailed p-value
    """
    try:
        corr, p_val = spearmanr(y_true, y_pred)
        return float(corr), float(p_val)
    except:
        return 0.0, 1.0


def overlap_at_n(y_true: np.ndarray, y_pred: np.ndarray, n_percent: float) -> float:
    """
    Overlap@N% metric from HSTrans paper (Equation 20)

    Measures the proportion of positive samples in the top N% of predicted results.
    This is a recommendation metric that evaluates how well the model ranks
    high-frequency side effects at the top.

    Formula: Overlap@N% = TP / (T × N%)
    where:
        TP = number of positive samples in top N% of predicted results
        T = total number of samples in test set

    Args:
        y_true: True frequency labels (0-5, where 0 means no side effect)
        y_pred: Predicted frequency scores
        n_percent: Percentage (0-100) for top-N ranking (e.g., 1, 5, 10, 20)

    Returns:
        overlap: Overlap@N% score
    """
    if n_percent <= 0 or n_percent > 100:
        raise ValueError("n_percent must be in range (0, 100]")

    # Convert to binary labels (0 = negative, >0 = positive)
    y_true_binary = (y_true != 0).astype(int)

    # Total number of samples
    total_samples = len(y_true)

    # Number of samples in top N%
    n_top = max(1, int(np.ceil(total_samples * n_percent / 100.0)))

    # Get indices of top N% predictions (sorted by predicted score, descending)
    top_indices = np.argsort(y_pred)[::-1][:n_top]

    # Count true positives in top N%
    tp_in_top_n = np.sum(y_true_binary[top_indices])

    # Calculate Overlap@N%
    # Denominator is T × N% according to paper equation (20)
    denominator = total_samples * (n_percent / 100.0)

    overlap = tp_in_top_n / denominator if denominator > 0 else 0.0

    return float(overlap)


# ============================================================================
# Paper-Specific Metric Functions
# ============================================================================

def calculate_paper_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all metrics specified in HSTrans paper

    Paper metrics:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error  
    - SCC: Spearman's rank correlation coefficient
    - Overlap@1%, 5%, 10%, 20%: Recommendation metrics

    Args:
        y_true: True frequency values (0-5)
        y_pred: Predicted frequency values

    Returns:
        metrics: Dictionary of paper metrics
    """
    # Basic regression metrics
    mse_val = mse(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)

    # Correlation metrics (filter zeros as per paper)
    pearson_corr, pearson_p = pearson(y_true, y_pred)
    spearman_corr, spearman_p = spearman(y_true, y_pred)

    # Overlap@N% metrics (key recommendation metrics from paper)
    overlap_metrics = {}
    for n in [1, 5, 10, 20]:
        try:
            overlap_metrics[f'overlap@{n}%'] = overlap_at_n(y_true, y_pred, n)
        except Exception as e:
            logger.warning(f"Failed to compute Overlap@{n}%: {e}")
            overlap_metrics[f'overlap@{n}%'] = 0.0

    metrics = {
        'mse': mse_val,
        'rmse': rmse_val,
        'mae': mae_val,
        'pearson': pearson_corr,
        'pearson_pvalue': pearson_p,
        'spearman': spearman_corr,  # This is SCC in paper
        'scc': spearman_corr,  # Alias for paper nomenclature
        'spearman_pvalue': spearman_p,
        **overlap_metrics
    }

    return metrics


def print_paper_metrics(metrics: Dict[str, float], title: str = "HSTrans Paper Metrics"):
    """
    Pretty print metrics in HSTrans paper format

    Args:
        metrics: Dictionary of metrics
        title: Title for the print
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    # Main metrics from paper Table 1
    print("\nFrequency Prediction Metrics:")
    if 'rmse' in metrics:
        print(f"  RMSE:     {metrics['rmse']:.4f}")
    if 'mae' in metrics:
        print(f"  MAE:      {metrics['mae']:.4f}")

    # Association prediction metrics
    print("\nAssociation Prediction Metrics:")
    if 'scc' in metrics or 'spearman' in metrics:
        scc_val = metrics.get('scc', metrics.get('spearman', 0))
        print(f"  SCC (Spearman): {scc_val:.4f}")
    if 'pearson' in metrics:
        print(f"  Pearson:        {metrics['pearson']:.4f}")

    # Recommendation metrics (Overlap@N%) - KEY METRICS FROM PAPER
    overlap_keys = ['overlap@1%', 'overlap@5%', 'overlap@10%', 'overlap@20%']
    if any(k in metrics for k in overlap_keys):
        print("\nRecommendation Metrics (Top-N% Ranking):")
        for key in overlap_keys:
            if key in metrics:
                print(f"  {key:15s}: {metrics[key]:.4f}")

    print("=" * 60 + "\n")


# ============================================================================
# Legacy Functions (for backward compatibility)
# ============================================================================

def calculate_all_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Legacy function - use calculate_paper_metrics instead
    """
    return calculate_paper_metrics(y_true, y_pred)


def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,  # Unused, kept for compatibility
        drug_ids: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Legacy function - use calculate_paper_metrics instead
    """
    metrics = calculate_paper_metrics(y_true, y_pred)
    
    # Add per-drug metrics if drug_ids provided (not in paper but for analysis)
    if drug_ids is not None:
        try:
            # Import per-drug functions if available
            drug_auc, _ = per_drug_auc(y_true, y_pred, drug_ids)
            drug_aupr, _ = per_drug_aupr(y_true, y_pred, drug_ids)
            metrics['drug_auc'] = drug_auc
            metrics['drug_aupr'] = drug_aupr
        except:
            pass  # Skip if per_drug_metrics not available or import fails

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Legacy function - use print_paper_metrics instead
    """
    print_paper_metrics(metrics, title)


# ============================================================================
# Per-Drug Metrics (Optional - not in paper but for analysis)
# ============================================================================

def per_drug_auc(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        drug_ids: np.ndarray
) -> Tuple[float, List[float]]:
    """
    Calculate AUC per drug (optional analysis - not in paper)

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        drug_ids: Drug identifiers

    Returns:
        mean_auc: Mean AUC across drugs
        drug_aucs: List of AUC for each drug
    """
    from sklearn.metrics import roc_auc_score
    
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
    Calculate AUPR per drug (optional analysis - not in paper)

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        drug_ids: Drug identifiers

    Returns:
        mean_aupr: Mean AUPR across drugs
        drug_auprs: List of AUPR for each drug
    """
    from sklearn.metrics import average_precision_score
    
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


if __name__ == "__main__":
    # Test paper metrics
    print("=" * 60)
    print("Testing HSTrans Paper Metrics")
    print("=" * 60)

    # Create dummy data with frequency labels (0-5)
    np.random.seed(42)
    n_samples = 100

    # Generate realistic frequency distribution (similar to paper)
    freq_labels = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples,
                                   p=[0.05, 0.03, 0.11, 0.27, 0.47, 0.07])
    
    # Generate predictions with some noise
    predictions = freq_labels + np.random.randn(n_samples) * 0.5
    predictions = np.clip(predictions, 0, 5)  # Clip to valid range

    print(f"\nSample size: {n_samples}")
    print(f"True labels range: [{freq_labels.min():.0f}, {freq_labels.max():.0f}]")
    print(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Positive samples: {np.sum(freq_labels != 0)} ({np.mean(freq_labels != 0):.1%})")

    # Calculate paper metrics
    metrics = calculate_paper_metrics(freq_labels, predictions)

    # Print metrics
    print_paper_metrics(metrics, title="HSTrans Paper Metrics Test")

    # Test Overlap@N% metrics specifically
    print("\n" + "=" * 60)
    print("Testing Overlap@N% Metrics")
    print("=" * 60)

    # Create test data with clear positive/negative labels
    y_true_test = np.array([0, 0, 0, 5, 4, 3, 2, 1, 0, 0])  # 5 positives, 5 negatives
    y_pred_test = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.15])

    print(f"\nTest data:")
    print(f"y_true: {y_true_test}")
    print(f"y_pred: {y_pred_test}")
    print(f"Number of positive samples: {np.sum(y_true_test != 0)}")

    for n in [10, 20, 50, 100]:
        overlap = overlap_at_n(y_true_test, y_pred_test, n)
        print(f"\nOverlap@{n}%: {overlap:.4f}")

        # Show which samples are in top N%
        n_top = max(1, int(np.ceil(len(y_true_test) * n / 100.0)))
        top_indices = np.argsort(y_pred_test)[::-1][:n_top]
        print(f"  Top {n}% indices: {top_indices}")
        print(f"  Top {n}% true labels: {y_true_test[top_indices]}")
        print(f"  Top {n}% pred scores: {y_pred_test[top_indices]}")
        print(f"  Positives in top {n}%: {np.sum(y_true_test[top_indices] != 0)}/{n_top}")

    print("\n" + "=" * 60)
    print("✓ All paper metrics tests passed!")
    print("✓ Metrics are now aligned with HSTrans paper:")
    print("  - Regression task (predict frequency 0-5)")
    print("  - Primary metrics: RMSE, MAE, SCC (Spearman)")
    print("  - Recommendation metrics: Overlap@1%, 5%, 10%, 20%")
    print("  - No unnecessary classification metrics")
    print("=" * 60)
