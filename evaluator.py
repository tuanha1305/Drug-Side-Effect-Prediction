"""
Evaluation module for drug side effect prediction
Regression task: Predict frequency scores (0-5) for drug-side effect pairs
Based on HSTrans paper evaluation framework
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from model import DrugSideEffectModel
from metrics import overlap_at_n

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for drug side effect prediction model
    Task: Regression - predict frequency scores (0-5)

    Evaluation metrics (from paper):
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - SCC: Spearman's rank correlation coefficient
    - Overlap@N%: Recommendation metric for top-N% predictions
    """

    def __init__(
            self,
            model: DrugSideEffectModel,
            device: str = 'cpu'
    ):
        """
        Initialize evaluator

        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
            self,
            data_loader: DataLoader,
            return_embeddings: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from model

        Args:
            data_loader: Data loader
            return_embeddings: Whether to return embeddings

        Returns:
            results: Dictionary with predictions, labels, and optionally embeddings
        """
        all_preds = []
        all_labels = []
        all_drug_embeddings = []
        all_se_embeddings = []

        for batch in tqdm(data_loader, desc="Predicting", leave=False):
            drug, se, drug_mask, se_mask, label = batch

            # Move to device
            drug = drug.to(self.device)
            se = se.to(self.device)
            drug_mask = drug_mask.to(self.device)
            se_mask = se_mask.to(self.device)
            label = label.to(self.device).float()

            # Forward pass
            output, drug_emb, se_emb = self.model(drug, se, drug_mask, se_mask)

            # Collect results
            all_preds.append(output.squeeze().cpu().numpy())
            all_labels.append(label.cpu().numpy())

            if return_embeddings:
                all_drug_embeddings.append(drug_emb.cpu().numpy())
                all_se_embeddings.append(se_emb.cpu().numpy())

        # Concatenate results
        results = {
            'predictions': np.concatenate(all_preds),
            'labels': np.concatenate(all_labels)
        }

        if return_embeddings:
            results['drug_embeddings'] = np.concatenate(all_drug_embeddings, axis=0)
            results['se_embeddings'] = np.concatenate(all_se_embeddings, axis=0)

        return results

    def evaluate_regression(
            self,
            predictions: np.ndarray,
            labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate regression metrics according to paper

        Metrics:
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - Pearson: Pearson correlation coefficient
        - Spearman (SCC): Spearman's rank correlation coefficient
        - Overlap@N%: Recommendation metrics for top N% predictions

        Args:
            predictions: Predicted frequency scores
            labels: True frequency labels (0-5)

        Returns:
            metrics: Dictionary of all evaluation metrics
        """
        # MSE and RMSE
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)

        # MAE
        mae = mean_absolute_error(labels, predictions)

        # Correlation metrics (filter out zeros for meaningful correlation)
        # According to paper, correlation is computed on non-zero samples
        valid_mask = labels != 0
        valid_preds = predictions[valid_mask]
        valid_labels = labels[valid_mask]

        pearson_corr = 0.0
        pearson_p = 1.0
        spearman_corr = 0.0
        spearman_p = 1.0

        if len(valid_labels) > 1 and len(np.unique(valid_labels)) > 1:
            try:
                pearson_corr, pearson_p = pearsonr(valid_labels, valid_preds)
            except:
                pass

            try:
                spearman_corr, spearman_p = spearmanr(valid_labels, valid_preds)
            except:
                pass

        # Overlap@N% metrics (recommendation metrics from paper)
        # These evaluate ranking quality for top-N% predictions
        overlap_metrics = {}
        for n in [1, 5, 10, 20]:
            try:
                overlap_metrics[f'overlap@{n}%'] = overlap_at_n(labels, predictions, n)
            except Exception as e:
                logger.warning(f"Failed to compute Overlap@{n}%: {e}")
                overlap_metrics[f'overlap@{n}%'] = 0.0

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'pearson': float(pearson_corr),
            'pearson_pvalue': float(pearson_p),
            'spearman': float(spearman_corr),  # This is SCC in paper
            'scc': float(spearman_corr),  # Alias for paper nomenclature
            'spearman_pvalue': float(spearman_p),
            **overlap_metrics
        }

        return metrics

    def evaluate_per_drug(
            self,
            predictions: np.ndarray,
            labels: np.ndarray,
            drug_ids: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate metrics per drug (average across drugs)

        Note: Paper does not mention per-drug metrics explicitly,
        but this can be useful for analysis.

        Args:
            predictions: Predicted frequency scores
            labels: True frequency labels
            drug_ids: Drug identifiers

        Returns:
            metrics: Dictionary of per-drug aggregated metrics
        """
        unique_drugs = np.unique(drug_ids)

        drug_rmse_scores = []
        drug_mae_scores = []
        drug_scc_scores = []

        for drug_id in unique_drugs:
            mask = drug_ids == drug_id
            drug_preds = predictions[mask]
            drug_labels = labels[mask]

            # Skip if insufficient data
            if len(drug_labels) < 2:
                continue

            # RMSE
            try:
                mse = mean_squared_error(drug_labels, drug_preds)
                rmse = np.sqrt(mse)
                drug_rmse_scores.append(rmse)
            except:
                pass

            # MAE
            try:
                mae = mean_absolute_error(drug_labels, drug_preds)
                drug_mae_scores.append(mae)
            except:
                pass

            # SCC (Spearman)
            valid_mask = drug_labels != 0
            if valid_mask.sum() > 1:
                try:
                    scc, _ = spearmanr(drug_labels[valid_mask], drug_preds[valid_mask])
                    if not np.isnan(scc):
                        drug_scc_scores.append(scc)
                except:
                    pass

        metrics = {
            'per_drug_rmse': float(np.mean(drug_rmse_scores)) if drug_rmse_scores else 0.0,
            'per_drug_mae': float(np.mean(drug_mae_scores)) if drug_mae_scores else 0.0,
            'per_drug_scc': float(np.mean(drug_scc_scores)) if drug_scc_scores else 0.0,
            'num_drugs_evaluated': len(drug_rmse_scores)
        }

        return metrics

    def evaluate(
            self,
            data_loader: DataLoader,
            evaluate_per_drug: bool = False,
            drug_ids: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Complete evaluation with all metrics from paper

        Metrics computed:
        - RMSE, MAE: Frequency prediction accuracy
        - Pearson, Spearman (SCC): Association prediction
        - Overlap@1%, 5%, 10%, 20%: Recommendation performance

        Args:
            data_loader: Data loader
            evaluate_per_drug: Whether to calculate per-drug metrics
            drug_ids: Drug IDs (required if evaluate_per_drug=True)

        Returns:
            metrics: Dictionary with all metrics
        """
        logger.info("Starting evaluation...")

        # Get predictions
        results = self.predict(data_loader)
        predictions = results['predictions']
        labels = results['labels']

        # Evaluate regression metrics (includes Overlap@N%)
        logger.info("Computing regression and ranking metrics...")
        all_metrics = self.evaluate_regression(predictions, labels)

        # Per-drug metrics (optional)
        if evaluate_per_drug and drug_ids is not None:
            logger.info("Computing per-drug metrics...")
            per_drug_metrics = self.evaluate_per_drug(predictions, labels, drug_ids)
            all_metrics.update(per_drug_metrics)

        # Add summary statistics
        all_metrics.update({
            'num_samples': len(predictions),
            'num_positive': int(np.sum(labels != 0)),
            'num_negative': int(np.sum(labels == 0)),
            'positive_ratio': float(np.mean(labels != 0))
        })

        logger.info("Evaluation completed")
        return all_metrics

    def print_metrics(self, metrics: Dict[str, float]):
        """
        Pretty print metrics according to paper format

        Paper metrics (Table 1):
        - RMSE, MAE, SCC (Spearman)
        - Overlap@1%, 5%, 10%, 20%

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 60)
        print("Evaluation Metrics (HSTrans Paper Format)")
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

        # Per-drug metrics (if computed)
        per_drug_keys = ['per_drug_rmse', 'per_drug_mae', 'per_drug_scc']
        if any(k in metrics for k in per_drug_keys):
            print("\nPer-Drug Metrics:")
            if 'per_drug_rmse' in metrics:
                print(f"  Avg RMSE:  {metrics['per_drug_rmse']:.4f}")
            if 'per_drug_mae' in metrics:
                print(f"  Avg MAE:   {metrics['per_drug_mae']:.4f}")
            if 'per_drug_scc' in metrics:
                print(f"  Avg SCC:   {metrics['per_drug_scc']:.4f}")
            if 'num_drugs_evaluated' in metrics:
                print(f"  Num drugs: {metrics['num_drugs_evaluated']}")

        # Dataset statistics
        if 'num_samples' in metrics:
            print("\nDataset Statistics:")
            print(f"  Total samples:  {metrics['num_samples']}")
            print(f"  Positive:       {metrics['num_positive']} ({metrics['positive_ratio']:.2%})")
            print(f"  Negative:       {metrics['num_negative']}")

        print("=" * 60 + "\n")

    def save_predictions(
            self,
            predictions: np.ndarray,
            labels: np.ndarray,
            output_path: str
    ):
        """
        Save predictions to file

        Args:
            predictions: Predicted frequency scores (continuous)
            labels: True frequency labels (0-5)
            output_path: Output file path
        """
        import pandas as pd

        df = pd.DataFrame({
            'prediction': predictions,
            'label': labels,
            'error': predictions - labels,
            'abs_error': np.abs(predictions - labels),
            'squared_error': (predictions - labels) ** 2
        })

        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    @staticmethod
    def aggregate_fold_results(
            fold_metrics: List[Dict[str, float]],
            display_format: str = 'paper'
    ) -> Dict[str, str]:
        """
        Aggregate results from multiple folds (5-fold CV as in paper)

        Paper format: "mean ± std" (e.g., "1.390 ± 0.009")

        Args:
            fold_metrics: List of metric dictionaries from each fold
            display_format: 'paper' for "mean ± std", 'dict' for separate mean/std

        Returns:
            aggregated: Dictionary with aggregated metrics
        """
        if len(fold_metrics) == 0:
            return {}

        # Get all metric keys
        metric_keys = set()
        for metrics in fold_metrics:
            metric_keys.update(metrics.keys())

        # Remove non-numeric keys
        non_numeric_keys = {'num_samples', 'num_positive', 'num_negative',
                            'positive_ratio', 'num_drugs_evaluated'}
        metric_keys = metric_keys - non_numeric_keys

        aggregated = {}

        for key in metric_keys:
            # Collect values across folds
            values = []
            for metrics in fold_metrics:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    values.append(float(metrics[key]))

            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)

                if display_format == 'paper':
                    # Paper format: "mean ± std"
                    aggregated[key] = f"{mean_val:.3f} ± {std_val:.3f}"
                else:
                    # Separate mean and std
                    aggregated[f'{key}_mean'] = float(mean_val)
                    aggregated[f'{key}_std'] = float(std_val)

        return aggregated

    @staticmethod
    def print_fold_results(
            aggregated_metrics: Dict[str, str],
            title: str = "5-Fold Cross-Validation Results"
    ):
        """
        Print aggregated fold results in paper format

        Args:
            aggregated_metrics: Aggregated metrics from aggregate_fold_results()
            title: Title for the output
        """
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

        # Main metrics from paper (Table 1 format)
        paper_keys = ['rmse', 'mae', 'scc', 'overlap@1%', 'overlap@5%',
                      'overlap@10%', 'overlap@20%']

        print("\nPaper Metrics (mean ± std):")
        print("-" * 60)

        for key in paper_keys:
            if key in aggregated_metrics:
                display_key = key.upper() if key in ['rmse', 'mae', 'scc'] else key
                print(f"  {display_key:15s}: {aggregated_metrics[key]}")

        # Other metrics
        other_keys = set(aggregated_metrics.keys()) - set(paper_keys)
        if other_keys:
            print("\nAdditional Metrics:")
            print("-" * 60)
            for key in sorted(other_keys):
                print(f"  {key:15s}: {aggregated_metrics[key]}")

        print("=" * 60 + "\n")


def compare_models(
        evaluators: List[Evaluator],
        data_loader: DataLoader,
        model_names: Optional[List[str]] = None
):
    """
    Compare multiple models

    Args:
        evaluators: List of evaluators
        data_loader: Data loader
        model_names: Names of models (optional)

    Returns:
        comparison_df: DataFrame with comparison
    """

    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(evaluators))]

    results = []

    for evaluator, name in zip(evaluators, model_names):
        logger.info(f"Evaluating {name}...")
        metrics = evaluator.evaluate(data_loader)
        metrics['model'] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df.set_index('model')

    return df


if __name__ == "__main__":
    # Test evaluator for regression task
    from config import get_default_config, ModelConfig
    from model import create_model
    from torch.utils.data import TensorDataset, DataLoader

    print("=" * 60)
    print("Testing Evaluator (Regression Task)")
    print("=" * 60)

    # Get config
    config = get_default_config()

    # Create model
    model_config = ModelConfig()
    model_config.vocab_size = 2586
    model = create_model(model_config, device=config.device)

    # Create dummy data with frequency labels (0-5)
    n_samples = 100
    seq_len = 50

    # Generate realistic frequency labels (0-5)
    freq_labels = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples,
                                   p=[0.05, 0.03, 0.11, 0.27, 0.47, 0.07])  # Similar to paper distribution

    dataset = TensorDataset(
        torch.randint(0, 2586, (n_samples, seq_len)),
        torch.randint(0, 2586, (n_samples, seq_len)),
        torch.ones((n_samples, seq_len)),
        torch.ones((n_samples, seq_len)),
        torch.tensor(freq_labels, dtype=torch.float32)
    )

    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Create evaluator
    print("\n1. Creating evaluator...")
    evaluator = Evaluator(model, device=config.device)
    print("✓ Evaluator created")

    # Test prediction
    print("\n2. Testing prediction...")
    results = evaluator.predict(data_loader, return_embeddings=True)
    print(f"Predictions shape: {results['predictions'].shape}")
    print(f"Labels shape: {results['labels'].shape}")
    print(f"Predictions range: [{results['predictions'].min():.2f}, {results['predictions'].max():.2f}]")
    print(f"Labels range: [{results['labels'].min():.0f}, {results['labels'].max():.0f}]")
    print(f"Drug embeddings shape: {results['drug_embeddings'].shape}")
    print(f"SE embeddings shape: {results['se_embeddings'].shape}")
    print("✓ Prediction works")

    # Test regression metrics
    print("\n3. Testing regression metrics (including Overlap@N%)...")
    reg_metrics = evaluator.evaluate_regression(
        results['predictions'],
        results['labels']
    )
    print("Regression metrics:")
    for key, value in reg_metrics.items():
        if 'pvalue' not in key:
            print(f"  {key:20s}: {value:.4f}")
    print("✓ Regression metrics work (includes Overlap@N%)")

    # Test complete evaluation
    print("\n4. Testing complete evaluation...")
    all_metrics = evaluator.evaluate(data_loader)
    evaluator.print_metrics(all_metrics)
    print("✓ Complete evaluation works")

    # Test save predictions
    print("\n5. Testing save predictions...")
    evaluator.save_predictions(
        results['predictions'],
        results['labels'],
        'test_predictions.csv'
    )
    print("✓ Save predictions works")

    # Test aggregate fold results (simulate 5-fold CV)
    print("\n6. Testing fold aggregation (5-fold CV simulation)...")

    # Simulate 5 folds with slightly different metrics
    fold_metrics = []
    for fold in range(5):
        # Add random noise to simulate different fold results
        noise = np.random.randn() * 0.01
        fold_result = {
            'rmse': 1.390 + noise,
            'mae': 0.725 + noise * 0.5,
            'scc': 0.527 + noise * 0.02,
            'overlap@1%': 0.129 + noise * 0.03,
            'overlap@5%': 0.463 + noise * 0.015,
            'overlap@10%': 0.436 + noise * 0.01,
            'overlap@20%': 0.579 + noise * 0.005,
        }
        fold_metrics.append(fold_result)
        print(
            f"Fold {fold}: RMSE={fold_result['rmse']:.3f}, MAE={fold_result['mae']:.3f}, SCC={fold_result['scc']:.3f}")

    # Aggregate results
    aggregated = Evaluator.aggregate_fold_results(fold_metrics, display_format='paper')
    Evaluator.print_fold_results(aggregated)
    print("✓ Fold aggregation works")

    print("\n" + "=" * 60)
    print("✓ All evaluator tests passed!")
    print("=" * 60)
    print("\nNote: This evaluator is now aligned with HSTrans paper:")
    print("  - Regression task (predict frequency 0-5)")
    print("  - Metrics: RMSE, MAE, SCC (Spearman)")
    print("  - Overlap@1%, 5%, 10%, 20% for ranking evaluation")
    print("  - No classification metrics (accuracy, precision, etc.)")
    print("=" * 60)