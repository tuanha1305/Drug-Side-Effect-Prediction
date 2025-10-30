"""
Evaluation module for drug side effect prediction - REGRESSION VERSION
Comprehensive metrics for regression tasks only
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from model import DrugSideEffectModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for drug side effect prediction model - REGRESSION ONLY
    Predicts continuous values (severity/intensity of side effects)
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
        Evaluate regression metrics

        Args:
            predictions: Predicted values
            labels: True labels

        Returns:
            metrics: Dictionary of regression metrics
        """
        # MSE and RMSE
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)

        # MAE
        mae = mean_absolute_error(labels, predictions)

        # R-squared
        try:
            r2 = r2_score(labels, predictions)
        except:
            r2 = 0.0

        # Correlation
        pearson_corr = 0.0
        spearman_corr = 0.0

        if len(np.unique(labels)) > 1 and len(np.unique(predictions)) > 1:
            try:
                pearson_corr, _ = pearsonr(labels, predictions)
            except:
                pass

            try:
                spearman_corr, _ = spearmanr(labels, predictions)
            except:
                pass

        # Additional metrics
        mean_abs_percentage_error = 0.0
        if np.all(np.abs(labels) > 1e-8):
            mean_abs_percentage_error = np.mean(np.abs((labels - predictions) / labels)) * 100

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'pearson': float(pearson_corr),
            'spearman': float(spearman_corr),
            'mape': float(mean_abs_percentage_error)
        }

        return metrics

    def evaluate_per_drug(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        drug_ids: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate REGRESSION metrics per drug (average across drugs)

        Args:
            predictions: Predicted values
            labels: True labels
            drug_ids: Drug identifiers

        Returns:
            metrics: Dictionary of per-drug regression metrics
        """
        unique_drugs = np.unique(drug_ids)

        drug_rmse_scores = []
        drug_mae_scores = []
        drug_pearson_scores = []
        drug_r2_scores = []

        for drug_id in unique_drugs:
            mask = drug_ids == drug_id
            drug_preds = predictions[mask]
            drug_labels = labels[mask]

            # Skip if too few samples
            if len(drug_labels) < 2:
                continue

            # RMSE
            try:
                rmse_val = np.sqrt(mean_squared_error(drug_labels, drug_preds))
                drug_rmse_scores.append(rmse_val)
            except:
                pass

            # MAE
            try:
                mae_val = mean_absolute_error(drug_labels, drug_preds)
                drug_mae_scores.append(mae_val)
            except:
                pass

            # R2
            try:
                r2_val = r2_score(drug_labels, drug_preds)
                drug_r2_scores.append(r2_val)
            except:
                pass

            # Pearson correlation
            try:
                if len(np.unique(drug_labels)) > 1:
                    corr, _ = pearsonr(drug_labels, drug_preds)
                    drug_pearson_scores.append(corr)
            except:
                pass

        metrics = {
            'drug_rmse': float(np.mean(drug_rmse_scores)) if drug_rmse_scores else 0.0,
            'drug_mae': float(np.mean(drug_mae_scores)) if drug_mae_scores else 0.0,
            'drug_r2': float(np.mean(drug_r2_scores)) if drug_r2_scores else 0.0,
            'drug_pearson': float(np.mean(drug_pearson_scores)) if drug_pearson_scores else 0.0,
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
        Complete evaluation with REGRESSION metrics only

        Args:
            data_loader: Data loader
            evaluate_per_drug: Whether to calculate per-drug metrics
            drug_ids: Drug IDs (required if evaluate_per_drug=True)

        Returns:
            metrics: Dictionary with all regression metrics
        """
        logger.info("Starting regression evaluation...")

        # Get predictions
        results = self.predict(data_loader)
        predictions = results['predictions']
        labels = results['labels']

        # Regression metrics
        logger.info("Computing regression metrics...")
        regression_metrics = self.evaluate_regression(predictions, labels)

        all_metrics = regression_metrics.copy()

        # Per-drug metrics
        if evaluate_per_drug and drug_ids is not None:
            logger.info("Computing per-drug regression metrics...")
            per_drug_metrics = self.evaluate_per_drug(predictions, labels, drug_ids)
            all_metrics.update(per_drug_metrics)

        # Add summary statistics for regression
        all_metrics.update({
            'num_samples': len(predictions),
            'mean_label': float(np.mean(labels)),
            'std_label': float(np.std(labels)),
            'min_label': float(np.min(labels)),
            'max_label': float(np.max(labels)),
            'mean_pred': float(np.mean(predictions)),
            'std_pred': float(np.std(predictions)),
            'min_pred': float(np.min(predictions)),
            'max_pred': float(np.max(predictions))
        })

        logger.info("Evaluation completed")
        return all_metrics

    def print_metrics(self, metrics: Dict[str, float]):
        """
        Pretty print REGRESSION metrics

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*70)
        print("REGRESSION EVALUATION METRICS")
        print("="*70)

        # Main regression metrics
        if 'rmse' in metrics:
            print("\nMain Regression Metrics:")
            print(f"  RMSE:             {metrics['rmse']:.6f}")
            print(f"  MSE:              {metrics.get('mse', 0):.6f}")
            print(f"  MAE:              {metrics['mae']:.6f}")
            print(f"  R²:               {metrics.get('r2', 0):.6f}")
            print(f"  Pearson Corr:     {metrics['pearson']:.6f}")
            print(f"  Spearman Corr:    {metrics['spearman']:.6f}")
            if metrics.get('mape', 0) > 0:
                print(f"  MAPE:             {metrics['mape']:.2f}%")

        # Per-drug regression metrics
        if 'drug_rmse' in metrics:
            print("\nPer-Drug Regression Metrics:")
            print(f"  Drug RMSE:        {metrics['drug_rmse']:.6f}")
            print(f"  Drug MAE:         {metrics['drug_mae']:.6f}")
            print(f"  Drug R²:          {metrics.get('drug_r2', 0):.6f}")
            print(f"  Drug Pearson:     {metrics['drug_pearson']:.6f}")
            print(f"  Drugs evaluated:  {metrics['num_drugs_evaluated']}")

        # Dataset statistics
        if 'num_samples' in metrics:
            print("\nDataset Statistics:")
            print(f"  Total samples:    {metrics['num_samples']}")
            if 'mean_label' in metrics:
                print(f"  Label stats:      {metrics['mean_label']:.4f} ± {metrics['std_label']:.4f}")
                print(f"  Label range:      [{metrics['min_label']:.4f}, {metrics['max_label']:.4f}]")
                print(f"  Pred stats:       {metrics['mean_pred']:.4f} ± {metrics['std_pred']:.4f}")
                print(f"  Pred range:       [{metrics['min_pred']:.4f}, {metrics['max_pred']:.4f}]")

        print("="*70 + "\n")

    def save_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        output_path: str
    ):
        """
        Save REGRESSION predictions to file

        Args:
            predictions: Predicted values
            labels: True labels
            output_path: Output file path
        """
        import pandas as pd

        df = pd.DataFrame({
            'prediction': predictions,
            'label': labels,
            'absolute_error': np.abs(predictions - labels),
            'squared_error': (predictions - labels) ** 2,
            'relative_error': np.abs(predictions - labels) / (np.abs(labels) + 1e-8),
            'error': predictions - labels
        })

        # Add percentile ranks
        df['prediction_percentile'] = df['prediction'].rank(pct=True) * 100
        df['label_percentile'] = df['label'].rank(pct=True) * 100

        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    def analyze_errors(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        percentiles: List[int] = [25, 50, 75, 90, 95, 99]
    ) -> Dict[str, float]:
        """
        Analyze error distribution

        Args:
            predictions: Predicted values
            labels: True labels
            percentiles: Percentiles to compute

        Returns:
            error_stats: Dictionary with error statistics
        """
        errors = predictions - labels
        abs_errors = np.abs(errors)

        error_stats = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'mean_abs_error': float(np.mean(abs_errors)),
            'median_abs_error': float(np.median(abs_errors)),
            'max_abs_error': float(np.max(abs_errors)),
            'min_abs_error': float(np.min(abs_errors))
        }

        # Percentiles
        for p in percentiles:
            error_stats[f'abs_error_p{p}'] = float(np.percentile(abs_errors, p))

        return error_stats


def compare_models(
    evaluators: List[Evaluator],
    data_loader: DataLoader,
    model_names: Optional[List[str]] = None
):
    """
    Compare multiple models on REGRESSION metrics

    Args:
        evaluators: List of evaluators
        data_loader: Data loader
        model_names: Names of models (optional)

    Returns:
        comparison_df: DataFrame with comparison
    """

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(evaluators))]

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
    # Test evaluator
    from config import get_default_config, ModelConfig
    from model import create_model
    from torch.utils.data import TensorDataset, DataLoader

    print("="*70)
    print("Testing Evaluator - REGRESSION VERSION")
    print("="*70)

    # Get config
    config = get_default_config()

    # Create model
    model_config = ModelConfig()
    model_config.vocab_size = 2586
    model = create_model(model_config, device=config.device)

    # Create dummy regression data
    n_samples = 100
    seq_len = 50

    # Labels are continuous values (e.g., side effect severity from 0 to 10)
    dataset = TensorDataset(
        torch.randint(0, 2586, (n_samples, seq_len)),
        torch.randint(0, 2586, (n_samples, seq_len)),
        torch.ones((n_samples, seq_len)),
        torch.ones((n_samples, seq_len)),
        torch.rand(n_samples) * 10  # Continuous labels [0, 10]
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
    print(f"Labels range: [{results['labels'].min():.2f}, {results['labels'].max():.2f}]")
    print("✓ Prediction works")

    # Test regression metrics
    print("\n3. Testing regression metrics...")
    reg_metrics = evaluator.evaluate_regression(
        results['predictions'],
        results['labels']
    )
    print("Regression metrics:")
    for key, value in reg_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("✓ Regression metrics work")

    # Test complete evaluation
    print("\n4. Testing complete evaluation...")
    all_metrics = evaluator.evaluate(data_loader)
    evaluator.print_metrics(all_metrics)
    print("✓ Complete evaluation works")

    # Test error analysis
    print("\n5. Testing error analysis...")
    error_stats = evaluator.analyze_errors(
        results['predictions'],
        results['labels']
    )
    print("Error statistics:")
    for key, value in error_stats.items():
        print(f"  {key}: {value:.4f}")
    print("✓ Error analysis works")

    # Test save predictions
    print("\n6. Testing save predictions...")
    evaluator.save_predictions(
        results['predictions'],
        results['labels'],
        'test_predictions_regression.csv'
    )
    print("✓ Save predictions works")

    print("\n" + "="*70)
    print("✓ All evaluator tests passed!")
    print("="*70)