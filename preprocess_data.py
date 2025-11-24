"""
Data preprocessing script for drug side effect prediction
Aligned with HSTrans paper.
Fixes:
1. Matches 'addition_negative_strategy' argument name in preprocessing.py.
2. Saves ALL negatives to processed_data.csv to allow dynamic balancing in dataset.py.
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
import sys

# Add current directory to path to ensure imports work
sys.path.append(str(Path(__file__).parent))

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
    """Load raw data files"""
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

    # Load drug SMILES
    drug_smiles_path = config.data.raw_data_dir / config.data.drug_smiles_file
    logger.info(f"Loading drug SMILES: {drug_smiles_path}")

    if not drug_smiles_path.exists():
        raise FileNotFoundError(f"Drug SMILES file not found: {drug_smiles_path}")

    drug_dict, drug_smiles = load_drug_smiles(str(drug_smiles_path))

    return drug_side_matrix, drug_dict, drug_smiles


def extract_samples(
    drug_side_matrix: np.ndarray,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract samples.
    CRITICAL: Must extract ALL negatives ('all' strategy) to allow
    dynamic sampling in dataset.py.
    """
    logger.info("\n" + "="*60)
    logger.info("Extracting Samples")
    logger.info("="*60)

    # FIXED: Argument name changed from 'strategy' to 'addition_negative_strategy'
    # to match preprocessing.py
    logger.info("Strategy: 'all' (Extracting all negatives for dynamic sampling)")

    addition_neg, final_pos, final_neg = extract_positive_negative_samples(
        drug_side_matrix,
        addition_negative_strategy='all'
    )

    logger.info(f"Positive samples: {len(final_pos)}")
    logger.info(f"Negative samples (balanced part): {len(final_neg)}")
    logger.info(f"Additional negative samples: {len(addition_neg)}")

    return addition_neg, final_pos, final_neg


def create_dataframe(
    addition_neg: np.ndarray,
    final_pos: np.ndarray,
    final_neg: np.ndarray,
    drug_smiles: List[str],
    save_path: Path
) -> pd.DataFrame:
    """Create and save DataFrame with ALL data"""
    logger.info("\n" + "="*60)
    logger.info("Creating DataFrame")
    logger.info("="*60)

    # Combine ALL samples: Positives + Balanced Negatives + Additional Negatives
    # This ensures processed_data.csv has the full pool of negatives
    all_negatives = np.vstack((final_neg, addition_neg))
    final_sample = np.vstack((final_pos, all_negatives))

    logger.info(f"Total samples for DataFrame: {len(final_sample)}")

    # Create DataFrame
    df = prepare_dataframes(final_sample, drug_smiles)

    logger.info(f"DataFrame shape: {df.shape}")

    # Save DataFrame
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved DataFrame to: {save_path}")

    return df


def create_cv_splits(
    df: pd.DataFrame,
    config: Config,
    save_path: Path
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create stratified cross-validation splits"""
    logger.info("\n" + "="*60)
    logger.info(f"Creating {config.data.n_folds}-Fold Cross-Validation Splits")
    logger.info("="*60)

    X = np.zeros(len(df)) # Placeholder
    # Stratify by Frequency Label (0, 1, 2, 3, 4, 5) to maintain distribution
    y = df['Label'].values.astype(int)

    skf = StratifiedKFold(
        n_splits=config.data.n_folds,
        random_state=config.data.random_state,
        shuffle=True
    )

    splits = list(skf.split(X, y))

    # Print fold statistics
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_pos = np.sum(df.iloc[train_idx]['Label'] != 0)
        val_pos = np.sum(df.iloc[val_idx]['Label'] != 0)
        # Note: Negatives will be huge here because we included all of them
        logger.info(f"Fold {fold_idx}: Train {len(train_idx)} (Pos: {train_pos}), Val {len(val_idx)} (Pos: {val_pos})")

    # Save splits
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(splits, f)
    logger.info(f"Saved CV splits to: {save_path}")

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
    Extract side effect substructure features.
    CRITICAL: Must use ONLY training data for each fold to avoid leakage.
    """
    logger.info("\n" + "="*60)
    logger.info("Extracting Side Effect Substructure Features")
    logger.info("="*60)
    logger.info("Note: Using training set of each fold to identify effective substructures.")

    # Create SMILES encoder (No cache needed here as we process sequentially)
    vocab_path = config.data.raw_data_dir / config.data.vocab_file
    subword_map_path = config.data.raw_data_dir / config.data.subword_map_file

    smiles_encoder = create_smiles_encoder(
        vocab_path=str(vocab_path),
        subword_map_path=str(subword_map_path),
        max_len=config.data.max_drug_len,
        use_cache=False
    )

    # Process each fold
    for fold_idx, (train_idx, _) in enumerate(tqdm(splits, desc="Processing folds")):

        # Output files for this fold
        index_file = output_dir / f"SE_sub_index_{config.data.top_k_substructures}_{fold_idx}.npy"
        mask_file = output_dir / f"SE_sub_mask_{config.data.top_k_substructures}_{fold_idx}.npy"

        if index_file.exists() and mask_file.exists():
            logger.info(f"Fold {fold_idx} features already exist. Skipping.")
            continue

        # Get training data for this fold
        # IMPORTANT: Even though df has all negatives, using stratify split ensures
        # train_idx has representative distribution. However, for effective substructure
        # mining, using too many negatives (class 0) might dilute the signal if not careful.
        # But the paper method (Chi-square/R-score) compares observed vs expected.
        # HSTrans typically balances BEFORE this step in paper, but practically,
        # using the full training distribution (which is imbalanced) is also valid
        # as long as we focus on the relationship between SE and Substructure.
        # To be safe and fast: We can filter train_df to be 1:1 balanced here if needed,
        # but the R-score method is robust. Let's use the full training set provided by split.

        train_df = df.iloc[train_idx].reset_index(drop=True)

        # Prepare data tuple list: (se_id, drug_smile, label)
        data = [
            (row.SE_id, row.Drug_smile, row.Label)
            for row in train_df.itertuples(index=False)
        ]

        # Extract side effect substructures
        se_index, se_mask = identify_side_effect_substructures(
            data=data,
            smiles_encoder=smiles_encoder,
            percentile_threshold=config.data.percentile_threshold, # 95.0
            top_k=config.data.top_k_substructures,
            fold=fold_idx,
            output_dir=output_dir
        )


def generate_statistics(
    df: pd.DataFrame,
    drug_side_matrix: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    output_path: Path
):
    """Generate and save dataset statistics"""
    logger.info("\n" + "="*60)
    logger.info("Generating Statistics")
    logger.info("="*60)

    stats = {
        'dataset': {
            'num_samples': len(df),
            'num_drugs': drug_side_matrix.shape[0],
            'num_side_effects': drug_side_matrix.shape[1],
            'positive_ratio': float(np.mean(df['Label'] != 0)),
        },
        'cross_validation': {
            'num_folds': len(splits)
        }
    }

    # Save statistics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Total samples: {stats['dataset']['num_samples']:,}")
    print(f"Positive ratio: {stats['dataset']['positive_ratio']:.2%}")


def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Preprocess HSTrans data')

    parser.add_argument('--data_dir', type=str, default='data/raw')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--top_k', type=int, default=50)

    # Default n_folds to 5 to match Paper
    parser.add_argument('--n_folds', type=int, default=5)

    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--skip_se_features', action='store_true')
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    # Configure
    config = get_default_config()
    config.data.raw_data_dir = Path(args.data_dir)
    config.data.processed_data_dir = Path(args.output_dir)
    config.data.top_k_substructures = args.top_k
    config.data.n_folds = args.n_folds
    config.data.random_state = args.random_state

    # Create output directory
    config.data.processed_data_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    df_path = config.data.processed_data_dir / "processed_data.csv"
    splits_path = config.data.processed_data_dir / "cv_splits.pkl"

    # Check existence
    if not args.force and df_path.exists() and splits_path.exists():
        logger.warning(f"Data already processed at {df_path}. Use --force to overwrite.")
        return

    # === Pipeline ===

    # 1. Load Raw Data
    drug_side_matrix, drug_dict, drug_smiles = load_raw_data(config)

    # 2. Extract Samples (ALL Negatives for dynamic balancing)
    addition_neg, final_pos, final_neg = extract_samples(drug_side_matrix, config)

    # 3. Create Master DataFrame (Merging all parts)
    df = create_dataframe(addition_neg, final_pos, final_neg, drug_smiles, df_path)

    # 4. Create 5-Fold Splits
    splits = create_cv_splits(df, config, splits_path)

    # 5. Extract SE Features (Per Fold to prevent leakage)
    if not args.skip_se_features:
        extract_se_features(
            drug_side_matrix,
            df,
            splits,
            drug_smiles,
            config,
            config.data.processed_data_dir
        )

    # 6. Stats
    stats_path = config.data.processed_data_dir / "dataset_statistics.json"
    generate_statistics(df, drug_side_matrix, splits, stats_path)

    logger.info("Preprocessing Complete.")


if __name__ == "__main__":
    main()