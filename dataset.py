"""
Dataset classes for drug side effect prediction
Aligned with HSTrans paper methodology:
- Regression task (Frequency 0-5) [cite: 110, 256]
- Balanced Negative Sampling (1:1 ratio)
- Substructure encoding [cite: 112]
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import logging

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugSideEffectDataset(Dataset):
    """
    Dataset for drug-side effect prediction
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        labels: np.ndarray,
        se_index: np.ndarray,
        se_mask: np.ndarray,
        smiles_encoder,
        fold: int = 0,
        cache_encoded: bool = True
    ):
        """
        Args:
            df: DataFrame with columns ['SE_id', 'Drug_smile', 'Label']
            indices: Indices to use from df
            labels: Labels array (Frequency scores)
            se_index: Pre-computed side effect substructure indices
            se_mask: Pre-computed side effect masks
            smiles_encoder: Function to encode SMILES strings
            fold: Fold number
            cache_encoded: Cache encoded SMILES for faster loading
        """
        self.df = df
        self.indices = indices
        self.labels = labels
        self.se_index = se_index.astype(np.int64)
        self.se_mask = se_mask.astype(np.int64)
        self.smiles_encoder = smiles_encoder
        self.fold = fold
        self.cache_encoded = cache_encoded

        # Cache for encoded SMILES
        self._smiles_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        if self.cache_encoded:
            self._precompute_encodings()

    def _precompute_encodings(self):
        """Pre-compute and cache all SMILES encodings"""
        # Only precompute for the drugs in the current split to save memory
        subset_df = self.df.iloc[self.indices]
        unique_smiles = subset_df['Drug_smile'].unique()

        count = 0
        for smile in unique_smiles:
            if smile not in self._smiles_cache:
                encoded, mask = self.smiles_encoder.encode(smile)
                self._smiles_cache[smile] = (encoded, mask)
                count += 1
        logger.info(f"Cached {count} unique SMILES encodings for fold {self.fold}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample
        Returns tensors formatted for HSTrans model
        """
        # Get actual index in dataframe
        data_idx = self.indices[idx]

        # Get drug SMILES and side effect ID
        row = self.df.iloc[data_idx]
        drug_smile = row['Drug_smile']
        se_id = int(row['SE_id'])

        # Get label (Frequency 0-5)
        # Paper uses regression (MSE), so label must be float [cite: 257]
        label_val = self.labels[data_idx]

        # Encode drug SMILES (Get drug substructures)
        if self.cache_encoded and drug_smile in self._smiles_cache:
            drug_encoded, drug_mask = self._smiles_cache[drug_smile]
        else:
            drug_encoded, drug_mask = self.smiles_encoder.encode(drug_smile)
            if self.cache_encoded:
                self._smiles_cache[drug_smile] = (drug_encoded, drug_mask)

        # Get side effect encoding (Get SE substructures)
        se_indices = self.se_index[se_id, :]
        se_mask_val = self.se_mask[se_id, :]

        # Convert to tensors
        # Indices must be Long for Embedding layers
        drug_encoded = torch.from_numpy(drug_encoded).long()
        se_indices = torch.from_numpy(se_indices).long()

        # Masks can be Long or Float, keeping Long for consistency
        drug_mask = torch.from_numpy(drug_mask).long()
        se_mask_val = torch.from_numpy(se_mask_val).long()

        # Label must be Float for MSELoss
        label = torch.tensor(float(label_val), dtype=torch.float32)

        return drug_encoded, se_indices, drug_mask, se_mask_val, label

    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics"""
        labels = self.labels[self.indices]
        return {
            'total_samples': len(self),
            'positive_samples': int(np.sum(labels > 0)),
            'negative_samples': int(np.sum(labels == 0)),
            'positive_ratio': float(np.mean(labels > 0))
        }


class DrugSideEffectDataModule:
    """
    Data module for handling data loading and preprocessing
    """

    def __init__(
        self,
        config: Config,
        smiles_encoder,
        fold: int = 0
    ):
        self.config = config
        self.data_config = config.data
        self.dataloader_config = config.dataloader
        self.smiles_encoder = smiles_encoder
        self.fold = fold

        self.train_dataset: Optional[DrugSideEffectDataset] = None
        self.val_dataset: Optional[DrugSideEffectDataset] = None

    def _balance_indices(self, indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Perform Negative Sampling as described in HSTrans paper.
        "To ensure a balance between positive and negative samples, we randomly
        selected 37,071 instances of class 0 data as negative samples."

        Args:
            indices: Array of indices (e.g., for training set)
            labels: Full labels array

        Returns:
            balanced_indices: Indices with 1:1 Positive:Negative ratio
        """
        # Get labels for these indices
        subset_labels = labels[indices]

        # Identify positive and negative indices within this subset
        pos_mask = subset_labels > 0
        neg_mask = subset_labels == 0

        pos_indices = indices[pos_mask]
        neg_indices = indices[neg_mask]

        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        if n_neg > n_pos:
            logger.info(f"Balancing data: Downsampling negatives from {n_neg} to {n_pos} (1:1 ratio)")
            # Randomly select negatives to match positives
            np.random.seed(self.config.training.seed + self.fold) # Ensure reproducibility per fold
            selected_neg_indices = np.random.choice(neg_indices, size=n_pos, replace=False)

            # Combine
            balanced_indices = np.concatenate([pos_indices, selected_neg_indices])
            np.random.shuffle(balanced_indices)
            return balanced_indices
        else:
            logger.info(f"Data already balanced or positives > negatives (Pos: {n_pos}, Neg: {n_neg})")
            return indices

    def setup(
        self,
        train_df: pd.DataFrame,
        train_indices: np.ndarray,
        train_labels: np.ndarray,
        val_df: pd.DataFrame,
        val_indices: np.ndarray,
        val_labels: np.ndarray,
        se_index: np.ndarray,
        se_mask: np.ndarray,
        balance_train: bool = True
    ):
        """
        Setup datasets with data splits

        Args:
            balance_train: Whether to balance training data (Default True for HSTrans)
        """
        logger.info(f"Setting up datasets for fold {self.fold}...")

        # === 1. Balance Training Data (Crucial for HSTrans) ===
        if balance_train:
            final_train_indices = self._balance_indices(train_indices, train_labels)
        else:
            final_train_indices = train_indices

        # Create training dataset
        self.train_dataset = DrugSideEffectDataset(
            df=train_df,
            indices=final_train_indices,
            labels=train_labels,
            se_index=se_index,
            se_mask=se_mask,
            smiles_encoder=self.smiles_encoder,
            fold=self.fold
        )

        # Create validation dataset (Validation usually reflects real distribution,
        # but paper implies CV on the balanced dataset[cite: 280].
        # If val_indices came from the already balanced split, _balance_indices won't do anything harm).
        self.val_dataset = DrugSideEffectDataset(
            df=val_df,
            indices=val_indices,
            labels=val_labels,
            se_index=se_index,
            se_mask=se_mask,
            smiles_encoder=self.smiles_encoder,
            fold=self.fold
        )

        train_stats = self.train_dataset.get_statistics()
        logger.info(f"Train set (Final): {train_stats['total_samples']} samples "
                   f"(Pos: {train_stats['positive_samples']}, Neg: {train_stats['negative_samples']})")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.dataloader_config.shuffle_train,
            num_workers=self.dataloader_config.num_workers,
            pin_memory=self.dataloader_config.pin_memory,
            persistent_workers=self.dataloader_config.persistent_workers
                if self.dataloader_config.num_workers > 0 else False,
            drop_last=self.dataloader_config.drop_last_train
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.dataloader_config.shuffle_val,
            num_workers=self.dataloader_config.num_workers,
            pin_memory=self.dataloader_config.pin_memory,
            persistent_workers=self.dataloader_config.persistent_workers
                if self.dataloader_config.num_workers > 0 else False,
            drop_last=self.dataloader_config.drop_last_val
        )

# Helper function preserved for compatibility
def create_dataloaders(
    config: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    se_index: np.ndarray,
    se_mask: np.ndarray,
    smiles_encoder,
    fold: int = 0
) -> Tuple[DataLoader, DataLoader]:

    data_module = DrugSideEffectDataModule(
        config=config,
        smiles_encoder=smiles_encoder,
        fold=fold
    )

    data_module.setup(
        train_df=train_df,
        train_indices=train_indices,
        train_labels=train_labels,
        val_df=val_df,
        val_indices=val_indices,
        val_labels=val_labels,
        se_index=se_index,
        se_mask=se_mask,
        balance_train=True # Enforce HSTrans requirement
    )

    return data_module.train_dataloader(), data_module.val_dataloader()