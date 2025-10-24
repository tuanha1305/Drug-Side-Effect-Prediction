"""
Main training script for drug side effect prediction
Complete pipeline from data loading to model training
"""

import argparse
import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging
import warnings
import json

from config import Config, get_default_config, get_fast_config, get_memory_efficient_config
from model import create_model
from dataset import create_dataloaders
from trainer import Trainer
from evaluator import Evaluator
from smiles_encoder import create_smiles_encoder, load_drug_smiles
from preprocessing import load_drug_side_matrix, extract_positive_negative_samples, prepare_dataframes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # Make CUDA operations deterministic (may impact performance)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(config: Config):
    """
    Load and prepare data

    Returns:
        df: DataFrame with all data
        drug_smiles: List of SMILES strings
        smiles_encoder: SMILES encoder instance
    """
    logger.info("=" * 60)
    logger.info("Loading Data")
    logger.info("=" * 60)

    # Load drug SMILES
    drug_smiles_path = config.data.raw_data_dir / config.data.drug_smiles_file
    logger.info(f"Loading drug SMILES from: {drug_smiles_path}")
    drug_dict, drug_smiles = load_drug_smiles(str(drug_smiles_path))

    # Create SMILES encoder
    vocab_path = config.data.raw_data_dir / config.data.vocab_file
    subword_map_path = config.data.raw_data_dir / config.data.subword_map_file

    logger.info("Creating SMILES encoder...")
    smiles_encoder = create_smiles_encoder(
        vocab_path=str(vocab_path),
        subword_map_path=str(subword_map_path),
        max_len=config.data.max_drug_len,
        use_cache=True
    )

    # Load drug-side effect matrix
    drug_side_pkl = config.data.raw_data_dir / config.data.drug_side_pkl
    logger.info(f"Loading drug-side effect matrix from: {drug_side_pkl}")
    drug_side_matrix = load_drug_side_matrix(str(drug_side_pkl))

    # Extract samples
    logger.info("Extracting positive and negative samples...")
    addition_neg, final_pos, final_neg = extract_positive_negative_samples(
        drug_side_matrix,
        config.data.addition_negative_strategy
    )

    # Combine samples
    final_sample = np.vstack((final_pos, final_neg))

    # Create DataFrame
    logger.info("Creating DataFrame...")
    df = prepare_dataframes(final_sample, drug_smiles)

    return df, drug_smiles, smiles_encoder


def load_cv_splits(config: Config):
    """Load cross-validation splits"""
    splits_path = config.data.processed_data_dir / "cv_splits.pkl"

    if splits_path.exists():
        logger.info(f"Loading CV splits from: {splits_path}")
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
    else:
        logger.info("Creating new CV splits...")
        from sklearn.model_selection import StratifiedKFold

        # Load data to create splits
        df, _, _ = load_data(config)
        X = df[['SE_id', 'Drug_smile']].values
        y = df['Label'].values

        skf = StratifiedKFold(
            n_splits=config.data.n_folds,
            random_state=config.data.random_state,
            shuffle=True
        )

        splits = list(skf.split(X, y))

        # Save splits
        config.data.processed_data_dir.mkdir(parents=True, exist_ok=True)
        with open(splits_path, 'wb') as f:
            pickle.dump(splits, f)
        logger.info(f"Saved CV splits to: {splits_path}")

    return splits


def load_side_effect_data(config: Config, fold: int):
    """Load side effect substructure data for a fold"""
    se_index_path = config.data.processed_data_dir / f"SE_sub_index_{config.data.top_k_substructures}_{fold}.npy"
    se_mask_path = config.data.processed_data_dir / f"SE_sub_mask_{config.data.top_k_substructures}_{fold}.npy"

    if not se_index_path.exists() or not se_mask_path.exists():
        logger.warning(f"Side effect data not found for fold {fold}")
        logger.warning("Creating dummy data. Run preprocessing.py first for real data!")
        se_index = np.random.randint(0, 2586, (994, 50))
        se_mask = np.ones((994, 50))
    else:
        logger.info(f"Loading side effect data for fold {fold}")
        se_index = np.load(se_index_path)
        se_mask = np.load(se_mask_path)

    return se_index, se_mask


def train_fold(
        config: Config,
        fold: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        se_index: np.ndarray,
        se_mask: np.ndarray,
        smiles_encoder
):
    """
    Train model for one fold

    Returns:
        trainer: Trained trainer instance
        metrics: Validation metrics
    """
    logger.info("=" * 60)
    logger.info(f"Training Fold {fold}")
    logger.info("=" * 60)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config=config,
        train_df=train_df,
        val_df=val_df,
        train_indices=train_indices,
        val_indices=val_indices,
        train_labels=train_labels,
        val_labels=val_labels,
        se_index=se_index,
        se_mask=se_mask,
        smiles_encoder=smiles_encoder,
        fold=fold
    )

    # Create model
    logger.info("Creating model...")
    model = create_model(config.model, device=config.device)

    if config.training.compile_model:
        print("Compiling model for inference only...")
        model = torch.compile(model, mode="reduce-overhead")

    # Count parameters
    param_counts = model.count_parameters()
    logger.info(f"Model parameters:")
    for key, value in param_counts.items():
        logger.info(f"  {key}: {value:,}")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        fold=fold
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Evaluate
    logger.info("Evaluating on validation set...")
    evaluator = Evaluator(model, device=config.device)
    metrics = evaluator.evaluate(val_loader)
    evaluator.print_metrics(metrics)

    # Save final results
    results_path = config.paths.result_dir / f"fold_{fold}_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved results to: {results_path}")

    return trainer, metrics


def train_all_folds(config: Config, args):
    """Train all folds"""
    logger.info("=" * 60)
    logger.info("Training All Folds")
    logger.info("=" * 60)

    # Load data
    df, drug_smiles, smiles_encoder = load_data(config)

    # Load CV splits
    splits = load_cv_splits(config)

    # Results storage
    all_metrics = []

    # Train each fold
    for fold_idx in range(args.start_fold, min(args.end_fold, len(splits))):
        train_indices, val_indices = splits[fold_idx]

        # Prepare fold data
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)

        train_labels = df.iloc[train_indices]['Label'].values
        val_labels = df.iloc[val_indices]['Label'].values

        # Load side effect data
        se_index, se_mask = load_side_effect_data(config, fold_idx)

        # Train fold
        trainer, metrics = train_fold(
            config=config,
            fold=fold_idx,
            train_df=train_df,
            val_df=val_df,
            train_indices=np.arange(len(train_df)),
            val_indices=np.arange(len(val_df)),
            train_labels=train_labels,
            val_labels=val_labels,
            se_index=se_index,
            se_mask=se_mask,
            smiles_encoder=smiles_encoder
        )

        all_metrics.append(metrics)

        logger.info(f"Completed fold {fold_idx}")
        logger.info("")

    # Aggregate results
    logger.info("=" * 60)
    logger.info("Cross-Validation Results")
    logger.info("=" * 60)

    # Calculate mean and std
    metric_keys = all_metrics[0].keys()
    aggregated = {}

    for key in metric_keys:
        if isinstance(all_metrics[0][key], (int, float)):
            values = [m[key] for m in all_metrics]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

    # Print aggregated results
    for key, value in aggregated.items():
        logger.info(f"{key}: {value:.4f}")

    # Save aggregated results
    agg_path = config.paths.result_dir / "aggregated_results.json"
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=4)
    logger.info(f"\nSaved aggregated results to: {agg_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train drug side effect prediction model')

    # Config selection
    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'fast', 'memory_efficient'],
                        help='Configuration preset')

    # Training parameters
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')

    # Cross-validation
    parser.add_argument('--start_fold', type=int, default=0,
                        help='Start fold index')
    parser.add_argument('--end_fold', type=int, default=10,
                        help='End fold index (exclusive)')

    # Optimization
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--compile_model', action='store_true',
                        help='Use torch.compile')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Paths
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Get config
    if args.config == 'default':
        config = get_default_config()
    elif args.config == 'fast':
        config = get_fast_config()
    elif args.config == 'memory_efficient':
        config = get_memory_efficient_config()

    # Override config with command line arguments
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.device is not None:
        config.device = args.device
    if args.use_amp:
        config.training.use_amp = True
    if args.compile_model:
        config.training.compile_model = True

    # Update paths
    config.data.raw_data_dir = Path(args.data_dir)
    config.paths.result_dir = Path(args.output_dir) / "results"
    config.paths.checkpoint_dir = Path(args.output_dir) / "checkpoints"
    config.paths.log_dir = Path(args.output_dir) / "logs"

    # Create directories
    config.paths.result_dir.mkdir(parents=True, exist_ok=True)
    config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.paths.log_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 60)
    logger.info("Configuration")
    logger.info("=" * 60)
    logger.info(f"Config preset: {args.config}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Epochs: {config.training.num_epochs}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Mixed precision: {config.training.use_amp}")
    logger.info(f"Compile model: {config.training.compile_model}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("")

    # Save config
    config_path = config.paths.result_dir / "config.json"
    config.save(str(config_path))
    logger.info(f"Saved configuration to: {config_path}\n")

    # Train
    train_all_folds(config, args)

    logger.info("=" * 60)
    logger.info("Training Completed Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()