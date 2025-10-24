"""
Evaluation module for drug side effect prediction
Comprehensive metrics for both regression and classification tasks
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error
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
    Evaluator for drug side effect prediction model
    Supports both regression and classification metrics
    """

    def __init__(
        self,
        model: DrugSideEffectModel,
        device: str = 'cpu',
        threshold: float = 0.5
    ):
        """
        Initialize evaluator

        Args:
            model: Model to evaluate
            device: Device to use
            threshold: Threshold for binary classification
        """
        self.model = model
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.threshold = threshold
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
        # Filter valid samples (non-zero labels for correlation)
        valid_mask = labels != 0
        valid_preds = predictions[valid_mask]
        valid_labels = labels[valid_mask]

        # MSE and RMSE
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)

        # MAE
        mae = mean_absolute_error(labels, predictions)

        # Correlation (only on valid samples)
        pearson = 0.0
        spearman = 0.0

        if len(valid_labels) > 1 and len(np.unique(valid_labels)) > 1:
            try:
                pearson, _ = pearsonr(valid_labels, valid_preds)
                spearman, _ = spearmanr(valid_labels, valid_preds)
            except:
                pass

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'pearson': float(pearson),
            'spearman': float(spearman)
        }

        return metrics

    def evaluate_classification(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate classification metrics

        Args:
            predictions: Predicted probabilities
            labels: True labels (0 or 1)
            threshold: Classification threshold (default: self.threshold)

        Returns:
            metrics: Dictionary of classification metrics
        """
        if threshold is None:
            threshold = self.threshold

        # Convert continuous predictions to binary
        pred_binary = (predictions > threshold).astype(int)

        # Convert labels to binary (non-zero -> 1)
        label_binary = (labels != 0).astype(int)

        # Basic metrics
        accuracy = accuracy_score(label_binary, pred_binary)
        precision = precision_score(label_binary, pred_binary, zero_division=0)
        recall = recall_score(label_binary, pred_binary, zero_division=0)
        f1 = f1_score(label_binary, pred_binary, zero_division=0)

        # AUC metrics (need probabilities)
        try:
            auc_roc = roc_auc_score(label_binary, predictions)
        except:
            auc_roc = 0.0

        try:
            auc_pr = average_precision_score(label_binary, predictions)
        except:
            auc_pr = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(label_binary, pred_binary).ravel()

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'specificity': float(specificity),
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
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

        Args:
            predictions: Predicted values
            labels: True labels
            drug_ids: Drug identifiers

        Returns:
            metrics: Dictionary of per-drug metrics
        """
        unique_drugs = np.unique(drug_ids)

        drug_auc_scores = []
        drug_aupr_scores = []

        for drug_id in unique_drugs:
            mask = drug_ids == drug_id
            drug_preds = predictions[mask]
            drug_labels = labels[mask]

            # Convert to binary
            drug_labels_binary = (drug_labels != 0).astype(int)

            # Skip if only one class present
            if len(np.unique(drug_labels_binary)) < 2:
                continue

            try:
                auc = roc_auc_score(drug_labels_binary, drug_preds)
                drug_auc_scores.append(auc)
            except:
                pass

            try:
                aupr = average_precision_score(drug_labels_binary, drug_preds)
                drug_aupr_scores.append(aupr)
            except:
                pass

        metrics = {
            'drug_auc': float(np.mean(drug_auc_scores)) if drug_auc_scores else 0.0,
            'drug_aupr': float(np.mean(drug_aupr_scores)) if drug_aupr_scores else 0.0,
            'num_drugs_evaluated': len(drug_auc_scores)
        }

        return metrics

    def evaluate(
        self,
        data_loader: DataLoader,
        evaluate_per_drug: bool = False,
        drug_ids: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Complete evaluation with all metrics

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

        # Regression metrics
        logger.info("Computing regression metrics...")
        regression_metrics = self.evaluate_regression(predictions, labels)

        # Classification metrics
        logger.info("Computing classification metrics...")
        classification_metrics = self.evaluate_classification(predictions, labels)

        # Combine metrics
        all_metrics = {**regression_metrics, **classification_metrics}

        # Per-drug metrics
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
        Pretty print metrics

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("Evaluation Metrics")
        print("="*60)

        # Regression metrics
        if 'rmse' in metrics:
            print("\nRegression Metrics:")
            print(f"  RMSE:     {metrics['rmse']:.4f}")
            print(f"  MAE:      {metrics['mae']:.4f}")
            print(f"  Pearson:  {metrics['pearson']:.4f}")
            print(f"  Spearman: {metrics['spearman']:.4f}")

        # Classification metrics
        if 'accuracy' in metrics:
            print("\nClassification Metrics:")
            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Recall:      {metrics['recall']:.4f}")
            print(f"  F1-Score:    {metrics['f1']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
            print(f"  AUC-PR:      {metrics['auc_pr']:.4f}")

        # Confusion matrix
        if 'tp' in metrics:
            print("\nConfusion Matrix:")
            print(f"  TP: {metrics['tp']:5d}  |  FP: {metrics['fp']:5d}")
            print(f"  FN: {metrics['fn']:5d}  |  TN: {metrics['tn']:5d}")

        # Per-drug metrics
        if 'drug_auc' in metrics:
            print("\nPer-Drug Metrics:")
            print(f"  Drug AUC:  {metrics['drug_auc']:.4f}")
            print(f"  Drug AUPR: {metrics['drug_aupr']:.4f}")
            print(f"  Num drugs: {metrics['num_drugs_evaluated']}")

        # Dataset info
        if 'num_samples' in metrics:
            print("\nDataset Statistics:")
            print(f"  Total samples:  {metrics['num_samples']}")
            print(f"  Positive:       {metrics['num_positive']} ({metrics['positive_ratio']:.2%})")
            print(f"  Negative:       {metrics['num_negative']}")

        print("="*60 + "\n")

    def save_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        output_path: str
    ):
        """
        Save predictions to file

        Args:
            predictions: Predicted values
            labels: True labels
            output_path: Output file path
        """
        import pandas as pd

        df = pd.DataFrame({
            'prediction': predictions,
            'label': labels,
            'error': np.abs(predictions - labels),
            'pred_binary': (predictions > self.threshold).astype(int),
            'label_binary': (labels != 0).astype(int)
        })

        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")


def compare_models(
    evaluators: List[Evaluator],
    data_loader: DataLoader,
    model_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple models

    Args:
        evaluators: List of evaluators
        data_loader: Data loader
        model_names: Names of models (optional)

    Returns:
        comparison_df: DataFrame with comparison
    """
    import pandas as pd

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

    print("="*60)
    print("Testing Evaluator")
    print("="*60)

    # Get config
    config = get_default_config()

    # Create model
    model_config = ModelConfig()
    model_config.vocab_size = 2586
    model = create_model(model_config, device=config.device)

    # Create dummy data
    n_samples = 100
    seq_len = 50

    dataset = TensorDataset(
        torch.randint(0, 2586, (n_samples, seq_len)),
        torch.randint(0, 2586, (n_samples, seq_len)),
        torch.ones((n_samples, seq_len)),
        torch.ones((n_samples, seq_len)),
        torch.rand(n_samples)
    )

    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Create evaluator
    print("\n1. Creating evaluator...")
    evaluator = Evaluator(model, device=config.device, threshold=0.5)
    print("✓ Evaluator created")

    # Test prediction
    print("\n2. Testing prediction...")
    results = evaluator.predict(data_loader, return_embeddings=True)
    print(f"Predictions shape: {results['predictions'].shape}")
    print(f"Labels shape: {results['labels'].shape}")
    print(f"Drug embeddings shape: {results['drug_embeddings'].shape}")
    print(f"SE embeddings shape: {results['se_embeddings'].shape}")
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

    # Test classification metrics
    print("\n4. Testing classification metrics...")
    cls_metrics = evaluator.evaluate_classification(
        results['predictions'],
        results['labels']
    )
    print("Classification metrics:")
    for key, value in cls_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("✓ Classification metrics work")

    # Test complete evaluation
    print("\n5. Testing complete evaluation...")
    all_metrics = evaluator.evaluate(data_loader)
    evaluator.print_metrics(all_metrics)
    print("✓ Complete evaluation works")

    # Test save predictions
    print("\n6. Testing save predictions...")
    evaluator.save_predictions(
        results['predictions'],
        results['labels'],
        'test_predictions.csv'
    )
    print("✓ Save predictions works")

    print("\n" + "="*60)
    print("✓ All evaluator tests passed!")
    print("="*60)