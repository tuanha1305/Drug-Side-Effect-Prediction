"""
Data preprocessing script for drug side effect prediction
Complete pipeline: Load data → Extract features → Create CV splits → Save processed data
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict, List
import logging
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import json

from config import Config, get_default_config
from preprocessing import (
    load_drug_side_matrix,
    extract_positive_negative_samples,
    prepare_dataframes,
    identify_side_effect_substructures
)
from smiles_encoder import load_drug_smiles, create_smiles_encoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(config: Config) -> Tuple[np.ndarray, Dict, List[str]]:
    """
    Load raw data files

    Args:
        config: Configuration object

    Returns:
        drug_side_matrix: Drug-side effect interaction matrix
        drug_dict: Dictionary mapping drug indices to ChEMBL IDs
        drug_smiles: List of SMILES strings
    """
    logger.info("="*60)
    logger.info("Loading Raw Data")
    logger.info("="*60)

    # Load drug-side effect matrix
    drug_side_pkl = config.data.raw_data_dir / config.data.drug_side_pkl
    logger.info(f"Loading drug-side effect matrix: {drug_side_pkl}")

    if not drug_side_pkl.exists():
        raise FileNotFoundError(f"Drug-side effect matrix not found: {drug_side_pkl}")

    drug_side_matrix = load_drug_side_matrix(str(drug_side_pkl))
    logger.info(f"Matrix shape: {drug_side_matrix.shape}")
    logger.info(f"Num drugs: {drug_side_matrix.shape[0]}")
    logger.info(f"Num side effects: {drug_side_matrix.shape[1]}")
    logger.info(f"Num interactions: {np.sum(drug_side_matrix > 0)}")

    # Load drug SMILES
    drug_smiles_path = config.data.raw_data_dir / config.data.drug_smiles_file
    logger.info(f"\nLoading drug SMILES: {drug_smiles_path}")

    if not drug_smiles_path.exists():
        raise FileNotFoundError(f"Drug SMILES file not found: {drug_smiles_path}")

    drug_dict, drug_smiles = load_drug_smiles(str(drug_smiles_path))
    logger.info(f"Num drugs with SMILES: {len(drug_smiles)}")

    return drug_side_matrix, drug_dict, drug_smiles


def extract_samples(
    drug_side_matrix: np.ndarray,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract positive and negative samples

    Args:
        drug_side_matrix: Drug-side effect interaction matrix
        config: Configuration object

    Returns:
        addition_neg: Additional negative samples
        final_pos: Final positive samples
        final_neg: Final negative samples
    """
    logger.info("\n" + "="*60)
    logger.info("Extracting Samples")
    logger.info("="*60)

    addition_neg, final_pos, final_neg = extract_positive_negative_samples(
        drug_side_matrix,
        config.data.addition_negative_strategy
    )

    logger.info(f"Positive samples: {len(final_pos)}")
    logger.info(f"Negative samples: {len(final_neg)}")
    logger.info(f"Additional negative samples: {len(addition_neg)}")
    logger.info(f"Total samples: {len(final_pos) + len(final_neg)}")
    logger.info(f"Positive ratio: {len(final_pos) / (len(final_pos) + len(final_neg)):.2%}")

    return addition_neg, final_pos, final_neg


def create_dataframe(
    final_pos: np.ndarray,
    final_neg: np.ndarray,
    drug_smiles: List[str],
    save_path: Path
) -> pd.DataFrame:
    """
    Create and save DataFrame

    Args:
        final_pos: Positive samples
        final_neg: Negative samples
        drug_smiles: List of SMILES strings
        save_path: Path to save DataFrame

    Returns:
        df: Complete DataFrame
    """
    logger.info("\n" + "="*60)
    logger.info("Creating DataFrame")
    logger.info("="*60)

    # Combine samples
    final_sample = np.vstack((final_pos, final_neg))
    logger.info(f"Combined samples shape: {final_sample.shape}")

    # Create DataFrame
    df = prepare_dataframes(final_sample, drug_smiles)

    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info("\nDataFrame info:")
    logger.info(f"  SE_id range: [{df['SE_id'].min()}, {df['SE_id'].max()}]")
    logger.info(f"  Label range: [{df['Label'].min():.2f}, {df['Label'].max():.2f}]")
    logger.info(f"  Unique SMILES: {df['Drug_smile'].nunique()}")
    logger.info(f"  Unique SEs: {df['SE_id'].nunique()}")

    # Save DataFrame
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"\nSaved DataFrame to: {save_path}")

    return df


def create_cv_splits(
    df: pd.DataFrame,
    config: Config,
    save_path: Path
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits

    Args:
        df: Complete DataFrame
        config: Configuration object
        save_path: Path to save splits

    Returns:
        splits: List of (train_indices, val_indices) tuples
    """
    logger.info("\n" + "="*60)
    logger.info("Creating Cross-Validation Splits")
    logger.info("="*60)

    X = df[['SE_id']].values  # Only SE_id, không có Drug_id
    y = (df['Label'] != 0).astype(int).values  # Binary labels for stratification

    skf = StratifiedKFold(
        n_splits=config.data.n_folds,
        random_state=config.data.random_state,
        shuffle=True
    )

    splits = list(skf.split(X, y))

    logger.info(f"Number of folds: {config.data.n_folds}")
    logger.info(f"Random state: {config.data.random_state}")

    # Print fold statistics
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_pos = np.sum(df.iloc[train_idx]['Label'] != 0)
        val_pos = np.sum(df.iloc[val_idx]['Label'] != 0)

        logger.info(f"\nFold {fold_idx}:")
        logger.info(f"  Train: {len(train_idx)} samples ({train_pos} positive, {train_pos/len(train_idx):.2%})")
        logger.info(f"  Val:   {len(val_idx)} samples ({val_pos} positive, {val_pos/len(val_idx):.2%})")

    # Save splits
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(splits, f)
    logger.info(f"\nSaved CV splits to: {save_path}")

    return splits


def extract_se_features(
    drug_side_matrix: np.ndarray,
    df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    drug_smiles: List[str],
    config: Config,
    output_dir: Path
):
    """
    Extract side effect substructure features for each fold

    Args:
        drug_side_matrix: Drug-side effect matrix
        df: Complete DataFrame
        splits: CV splits
        drug_smiles: List of SMILES strings
        config: Configuration object
        output_dir: Output directory
    """
    logger.info("\n" + "="*60)
    logger.info("Extracting Side Effect Substructure Features")
    logger.info("="*60)

    # Create SMILES encoder
    vocab_path = config.data.raw_data_dir / config.data.vocab_file
    subword_map_path = config.data.raw_data_dir / config.data.subword_map_file

    logger.info("Creating SMILES encoder...")
    smiles_encoder = create_smiles_encoder(
        vocab_path=str(vocab_path),
        subword_map_path=str(subword_map_path),
        max_len=config.data.max_drug_len,
        use_cache=False  # Don't use cache for preprocessing
    )

    # Process each fold
    for fold_idx, (train_idx, _) in enumerate(tqdm(splits, desc="Processing folds")):
        logger.info(f"\nProcessing fold {fold_idx}...")

        # Get training data for this fold
        train_df = df.iloc[train_idx].reset_index(drop=True)

        # Prepare data for identify_side_effect_substructures
        # Format: List of (se_id, drug_smile, label) tuples
        data = []
        for _, row in train_df.iterrows():
            data.append((row['SE_id'], row['Drug_smile'], row['Label']))

        # Extract side effect substructures
        logger.info(f"  Identifying substructures (top_k={config.data.top_k_substructures})...")
        se_index, se_mask = identify_side_effect_substructures(
            data=data,
            smiles_encoder=smiles_encoder,
            percentile_threshold=95.0,
            top_k=config.data.top_k_substructures,
            fold=fold_idx,
            output_dir=output_dir
        )

        logger.info(f"  SE index shape: {se_index.shape}")
        logger.info(f"  SE mask shape: {se_mask.shape}")
        logger.info(f"  Saved to: {output_dir}")


def generate_statistics(
    df: pd.DataFrame,
    drug_side_matrix: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    output_path: Path
):
    """
    Generate and save dataset statistics

    Args:
        df: Complete DataFrame
        drug_side_matrix: Drug-side effect matrix
        splits: CV splits
        output_path: Path to save statistics
    """
    logger.info("\n" + "="*60)
    logger.info("Generating Statistics")
    logger.info("="*60)

    stats = {
        'dataset': {
            'num_samples': len(df),
            'num_drugs': drug_side_matrix.shape[0],
            'num_side_effects': drug_side_matrix.shape[1],
            'num_interactions': int(np.sum(drug_side_matrix > 0)),
            'num_positive': int(np.sum(df['Label'] != 0)),
            'num_negative': int(np.sum(df['Label'] == 0)),
            'positive_ratio': float(np.mean(df['Label'] != 0)),
            'label_mean': float(df['Label'].mean()),
            'label_std': float(df['Label'].std()),
            'label_min': float(df['Label'].min()),
            'label_max': float(df['Label'].max())
        },
        'drugs': {
            'unique_smiles': int(df['Drug_smile'].nunique()),
            'samples_per_smiles_mean': float(df.groupby('Drug_smile').size().mean()),
            'samples_per_smiles_std': float(df.groupby('Drug_smile').size().std())
        },
        'side_effects': {
            'unique_side_effects': int(df['SE_id'].nunique()),
            'samples_per_se_mean': float(df.groupby('SE_id').size().mean()),
            'samples_per_se_std': float(df.groupby('SE_id').size().std())
        },
        'cross_validation': {
            'num_folds': len(splits),
            'folds': []
        }
    }

    # Fold statistics
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_stats = {
            'fold': fold_idx,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'train_positive': int(np.sum(df.iloc[train_idx]['Label'] != 0)),
            'val_positive': int(np.sum(df.iloc[val_idx]['Label'] != 0)),
            'train_pos_ratio': float(np.mean(df.iloc[train_idx]['Label'] != 0)),
            'val_pos_ratio': float(np.mean(df.iloc[val_idx]['Label'] != 0))
        }
        stats['cross_validation']['folds'].append(fold_stats)

    # Save statistics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Saved statistics to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("Dataset Statistics Summary")
    print("="*60)
    print(f"Total samples:        {stats['dataset']['num_samples']:,}")
    print(f"Drugs:                {stats['dataset']['num_drugs']:,}")
    print(f"Side effects:         {stats['dataset']['num_side_effects']:,}")
    print(f"Interactions:         {stats['dataset']['num_interactions']:,}")
    print(f"Positive samples:     {stats['dataset']['num_positive']:,} ({stats['dataset']['positive_ratio']:.2%})")
    print(f"Negative samples:     {stats['dataset']['num_negative']:,}")
    print(f"\nLabel statistics:")
    print(f"  Mean:  {stats['dataset']['label_mean']:.4f}")
    print(f"  Std:   {stats['dataset']['label_std']:.4f}")
    print(f"  Range: [{stats['dataset']['label_min']:.4f}, {stats['dataset']['label_max']:.4f}]")
    print("="*60)


def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Preprocess drug side effect data')

    # Paths
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Raw data directory')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')

    # Preprocessing options
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top K substructures per side effect')
    parser.add_argument('--n_folds', type=int, default=10,
                       help='Number of CV folds')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')

    # Processing options
    parser.add_argument('--skip_se_features', action='store_true',
                       help='Skip side effect feature extraction (faster)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if files exist')

    args = parser.parse_args()

    # Get config
    config = get_default_config()
    config.data.raw_data_dir = Path(args.data_dir)
    config.data.processed_data_dir = Path(args.output_dir)
    config.data.top_k_substructures = args.top_k
    config.data.n_folds = args.n_folds
    config.data.random_state = args.random_state

    logger.info("="*60)
    logger.info("Data Preprocessing Pipeline")
    logger.info("="*60)
    logger.info(f"Raw data dir:      {config.data.raw_data_dir}")
    logger.info(f"Output dir:        {config.data.processed_data_dir}")
    logger.info(f"Top K:             {config.data.top_k_substructures}")
    logger.info(f"CV folds:          {config.data.n_folds}")
    logger.info(f"Random state:      {config.data.random_state}")
    logger.info(f"Skip SE features:  {args.skip_se_features}")

    # Create output directory
    config.data.processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    df_path = config.data.processed_data_dir / "processed_data.csv"
    splits_path = config.data.processed_data_dir / "cv_splits.pkl"

    if not args.force and df_path.exists() and splits_path.exists():
        logger.warning(f"\nProcessed data already exists at: {df_path}")
        logger.warning("Use --force to reprocess")

        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Exiting...")
            return

    # Step 1: Load raw data
    drug_side_matrix, drug_dict, drug_smiles = load_raw_data(config)

    # Step 2: Extract samples
    addition_neg, final_pos, final_neg = extract_samples(drug_side_matrix, config)

    # Step 3: Create DataFrame
    df = create_dataframe(final_pos, final_neg, drug_smiles, df_path)

    # Step 4: Create CV splits
    splits = create_cv_splits(df, config, splits_path)

    # Step 5: Extract side effect features (optional)
    if not args.skip_se_features:
        extract_se_features(
            drug_side_matrix,
            df,
            splits,
            drug_smiles,
            config,
            config.data.processed_data_dir
        )
    else:
        logger.info("\n" + "="*60)
        logger.info("Skipping side effect feature extraction")
        logger.info("Run without --skip_se_features to extract features")
        logger.info("="*60)

    # Step 6: Generate statistics
    stats_path = config.data.processed_data_dir / "dataset_statistics.json"
    generate_statistics(df, drug_side_matrix, splits, stats_path)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Preprocessing Complete!")
    logger.info("="*60)
    logger.info(f"\nOutput files:")
    logger.info(f"  ✓ DataFrame:       {df_path}")
    logger.info(f"  ✓ CV splits:       {splits_path}")
    logger.info(f"  ✓ Statistics:      {stats_path}")

    if not args.skip_se_features:
        logger.info(f"  ✓ SE features:     {config.data.processed_data_dir / 'SE_sub_*'}")

    logger.info(f"\nNext steps:")
    logger.info(f"  1. python train.py")
    logger.info(f"  2. tensorboard --logdir logs/tensorboard")
    logger.info(f"  3. python evaluate.py")


if __name__ == "__main__":
    main()