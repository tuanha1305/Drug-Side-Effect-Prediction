"""
Evaluation script for drug side effect prediction
Executes 5-fold cross-validation evaluation as per HSTrans paper
"""

import argparse
import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
import json
import torch.serialization

from config import Config
from model import create_model
from evaluator import Evaluator
from dataset import DrugSideEffectDataset
from torch.utils.data import DataLoader
from smiles_encoder import create_smiles_encoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Config:
    """Load configuration from file"""
    logger.info(f"Loading config from: {config_path}")
    config = Config.load(config_path)
    return config


def load_model_from_checkpoint(
        checkpoint_path: str,
        config: Config,
        device: str = 'cpu'
) -> torch.nn.Module:
    """
    Load model from checkpoint
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    # Create model structure
    model = create_model(config.model, device=device)

    # Load checkpoint weights
    # Note: add_safe_globals is for newer PyTorch/Numpy compatibility
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except:
        pass # Ignore if older pytorch version

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        best_metric = checkpoint.get('best_metric', 'unknown')
        logger.info(f"Loaded model from epoch: {epoch}, Best metric: {best_metric}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded model state dict directly")

    model.eval()
    return model


def load_test_data(
        config: Config,
        fold: int,
        smiles_encoder
):
    """
    Load test data for a specific fold (part of 5-fold CV)
    """
    logger.info(f"Loading test data for fold {fold}...")

    # Load CV splits
    splits_path = config.data.processed_data_dir / "cv_splits.pkl"
    if not splits_path.exists():
        raise FileNotFoundError(f"CV splits not found: {splits_path}. Run preprocessing first.")

    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)

    # Load full dataset dataframe
    processed_data_path = config.data.processed_data_dir / "processed_data.csv"
    if not processed_data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {processed_data_path}")

    df = pd.read_csv(processed_data_path)

    # Get test indices for this fold
    # splits[fold] is (train_idx, test_idx)
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range. Total splits: {len(splits)}")

    _, test_indices = splits[fold]
    test_df = df.iloc[test_indices].reset_index(drop=True)
    test_labels = test_df['Label'].values

    # Load side effect data (pre-computed substructures)
    # Filenames match config logic
    se_index_path = config.data.processed_data_dir / f"SE_sub_index_{config.data.top_k_substructures}_{fold}.npy"
    se_mask_path = config.data.processed_data_dir / f"SE_sub_mask_{config.data.top_k_substructures}_{fold}.npy"

    if not se_index_path.exists() or not se_mask_path.exists():
        raise FileNotFoundError(f"Side effect data not found for fold {fold}")

    se_index = np.load(se_index_path)
    se_mask = np.load(se_mask_path)

    # Create dataset
    test_dataset = DrugSideEffectDataset(
        df=test_df,
        indices=np.arange(len(test_df)),
        labels=test_labels,
        se_index=se_index,
        se_mask=se_mask,
        smiles_encoder=smiles_encoder,
        fold=fold,
        cache_encoded=True
    )

    # Create dataloader
    # Note: Shuffle=False for evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory
    )

    logger.info(f"Test set (Fold {fold}): {len(test_dataset)} samples")

    return test_loader, test_df


def evaluate_fold(
        model,
        test_loader,
        config: Config,
        fold: int,
        save_predictions: bool = True
):
    """
    Evaluate model on a single fold
    """
    logger.info(f"Evaluating fold {fold}...")

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=config.device
    )

    # Evaluate
    # Note: evaluate() inside Evaluator now correctly handles SCC calculation (keeping zeros)
    metrics = evaluator.evaluate(test_loader)

    # Print metrics for this fold
    evaluator.print_metrics(metrics)

    # Save metrics
    results_dir = config.paths.result_dir / f"fold_{fold}"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved metrics to: {metrics_path}")

    # Save predictions (useful for analysis)
    if save_predictions:
        results = evaluator.predict(test_loader)
        predictions_path = results_dir / "predictions.csv"

        evaluator.save_predictions(
            results['predictions'],
            results['labels'],
            str(predictions_path)
        )
        logger.info(f"Saved predictions to: {predictions_path}")

    return metrics


def evaluate_all_folds(args):
    """
    Evaluate all folds and aggregate results.
    Paper Section 4.3: "averaging the results obtained across all 5 folds"
    """
    logger.info("=" * 60)
    logger.info("Starting Evaluation (5-Fold Cross-Validation)")
    logger.info("=" * 60)

    # Load config
    config_path = Path(args.checkpoint_dir) / "config.json"
    if args.config_path:
         config_path = Path(args.config_path)

    if not config_path.exists():
        # Fallback to default output location
        config_path = Path(args.output_dir) / "results" / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}. Please provide --config_path")

    config = load_config(str(config_path))

    # Override device if specified
    if args.device:
        config.device = args.device

    # Create SMILES encoder (shared across folds)
    vocab_path = config.data.raw_data_dir / config.data.vocab_file
    subword_map_path = config.data.raw_data_dir / config.data.subword_map_file

    logger.info("Creating SMILES encoder...")
    smiles_encoder = create_smiles_encoder(
        vocab_path=str(vocab_path),
        subword_map_path=str(subword_map_path),
        max_len=config.data.max_drug_len,
        use_cache=True
    )

    # Evaluate each fold
    all_metrics = []

    # Paper uses 5 folds
    for fold in range(args.start_fold, args.end_fold):
        logger.info("\n" + "-" * 60)
        logger.info(f"Processing Fold {fold}")
        logger.info("-" * 60)

        # Find checkpoint for this fold
        # Structure assumed: checkpoint_dir/fold_X/best_model.pth
        checkpoint_path = Path(args.checkpoint_dir) / f"fold_{fold}" / args.checkpoint_name

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            logger.warning(f"Skipping fold {fold}")
            continue

        try:
            # Load model
            model = load_model_from_checkpoint(
                str(checkpoint_path),
                config,
                device=config.device
            )

            # Load test data
            test_loader, test_df = load_test_data(
                config,
                fold,
                smiles_encoder
            )

            # Evaluate
            metrics = evaluate_fold(
                model=model,
                test_loader=test_loader,
                config=config,
                fold=fold,
                save_predictions=args.save_predictions
            )

            all_metrics.append(metrics)

        except Exception as e:
            logger.error(f"Error evaluating fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results across folds
    if len(all_metrics) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("Aggregated Results (Mean ± Std)")
        logger.info("=" * 60)

        # Use Evaluator's aggregation method for consistency
        aggregated_display = Evaluator.aggregate_fold_results(all_metrics, display_format='paper')
        Evaluator.print_fold_results(aggregated_display)

        # Also get raw mean/std values for saving to JSON
        aggregated_raw = Evaluator.aggregate_fold_results(all_metrics, display_format='dict')

        # Save aggregated results
        output_dir = Path(args.output_dir) / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        agg_path = output_dir / "final_aggregated_results.json"
        with open(agg_path, 'w') as f:
            json.dump(aggregated_raw, f, indent=4)

        logger.info(f"\nSaved aggregated results to: {agg_path}")

        # Verify specific paper metrics are present
        if 'overlap@1%_mean' in aggregated_raw:
             logger.info(f"Overlap@1%: {aggregated_raw['overlap@1%_mean']:.3f} (Compare with Paper Table 1)")

    else:
        logger.warning("No folds were successfully evaluated")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained HSTrans model')

    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing fold checkpoints (e.g., checkpoints/fold_0/)')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pth',
                        help='Checkpoint filename to load')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for evaluation results')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config file (optional, defaults to searching in checkpoint_dir)')

    # Evaluation settings
    # FIXED: Default end_fold set to 5 to match Paper
    parser.add_argument('--start_fold', type=int, default=0,
                        help='Start fold index')
    parser.add_argument('--end_fold', type=int, default=5,
                        help='End fold index (exclusive, default 5 for 5-fold CV)')

    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save detailed predictions to CSV for each fold')

    # Single fold evaluation
    parser.add_argument('--fold', type=int, default=None,
                        help='Evaluate single fold only (overrides start/end fold)')

    args = parser.parse_args()

    # Single fold mode override
    if args.fold is not None:
        args.start_fold = args.fold
        args.end_fold = args.fold + 1

    # Execute
    evaluate_all_folds(args)

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Completed Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()