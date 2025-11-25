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
    - RMSE: Root Mean Squared Error [cite: 287]
    - MAE: Mean Absolute Error [cite: 287]
    - SCC: Spearman's rank correlation coefficient [cite: 292]
    - Overlap@N%: Recommendation metric for top-N% predictions [cite: 296]
    """

    def __init__(
            self,
            model: DrugSideEffectModel,
            device: str = 'cpu'
    ):
        """
        Initialize evaluator
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
        Evaluate regression metrics according to paper.

        CORRECTION:
        SCC (Spearman) is calculated on the full test set N (Equation 19),
        which includes both positive and negative (0) samples[cite: 279, 294].
        """

        # =========================================================
        # [DEBUG CODE] BẮT ĐẦU: Kiểm tra phân phối dữ liệu
        # =========================================================
        print("\n" + "!"*50)
        print("DEBUG: KIỂM TRA DỰ ĐOÁN (PREDICTIONS CHECK)")
        print(f"Sample Size: {len(predictions)}")
        
        # 1. Xem 20 mẫu đầu tiên để so sánh trực quan
        print(f"\nTop 20 True Labels: {labels[:20]}")
        print(f"Top 20 Predictions: {predictions[:20]}")
        
        # 2. Kiểm tra thống kê (Quan trọng để bắt lỗi Collapsing)
        print(f"\nThống kê Labels (Thực tế):")
        print(f"  Min: {labels.min():.4f} | Max: {labels.max():.4f} | Mean: {labels.mean():.4f}")
        
        print(f"Thống kê Preds  (Dự đoán):")
        print(f"  Min: {predictions.min():.4f} | Max: {predictions.max():.4f}")
        print(f"  Mean: {predictions.mean():.4f} | Std (Độ lệch chuẩn): {predictions.std():.4f}")
        
        if predictions.std() < 0.01:
            print("\n>>> CẢNH BÁO: Std quá thấp! Mô hình đang bị 'Collapse' (Dự đoán toàn bộ giống nhau).")
            print(">>> Nguyên nhân có thể: Learning Rate quá lớn, hoặc dữ liệu Train chưa cân bằng.")
        else:
            print("\n>>> TRẠNG THÁI: Std ổn định. Mô hình có sự phân biệt giữa các mẫu.")
            
        print("!"*50 + "\n")
        # =========================================================
        # [DEBUG CODE] KẾT THÚC
        # =========================================================

        # MSE and RMSE [cite: 287]
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)

        # MAE [cite: 287]
        mae = mean_absolute_error(labels, predictions)

        # Correlation metrics
        # FIXED: Calculated on ALL samples (including class 0) to match
        # the definition of N in Equation 19.
        pearson_corr = 0.0
        pearson_p = 1.0
        spearman_corr = 0.0
        spearman_p = 1.0

        # Pearson
        try:
            pearson_corr, pearson_p = pearsonr(labels, predictions)
        except Exception:
            pass

        # Spearman (SCC) [cite: 292]
        try:
            spearman_corr, spearman_p = spearmanr(labels, predictions)
        except Exception:
            pass

        # Overlap@N% metrics (recommendation metrics from paper) [cite: 296]
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

    def evaluate(
            self,
            data_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Complete evaluation with all metrics from paper
        """
        logger.info("Starting evaluation...")

        # Get predictions
        results = self.predict(data_loader)
        predictions = results['predictions']
        labels = results['labels']

        # Evaluate regression metrics (includes Overlap@N%)
        logger.info("Computing regression and ranking metrics...")
        all_metrics = self.evaluate_regression(predictions, labels)

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
        Pretty print metrics according to paper format (Table 1)
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
        Paper format: "mean ± std"
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