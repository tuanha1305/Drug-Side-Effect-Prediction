"""
Dataset classes for drug side effect prediction
Optimized for PyTorch 2.x with caching and efficient data loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pickle
from functools import lru_cache
import logging

from config import Config, DataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugSideEffectDataset(Dataset):
    """
    Dataset for drug-side effect prediction
    Optimized for PyTorch 2.x with efficient data loading
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
            labels: Labels array
            se_index: Pre-computed side effect substructure indices [994, 50]
            se_mask: Pre-computed side effect masks [994, 50]
            smiles_encoder: Function to encode SMILES strings
            fold: Fold number for cross-validation
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
        
        # Pre-encode all SMILES if caching is enabled
        if self.cache_encoded:
            self._precompute_encodings()
    
    def _precompute_encodings(self):
        """Pre-compute and cache all SMILES encodings"""
        logger.info(f"Pre-computing SMILES encodings for fold {self.fold}...")
        unique_smiles = self.df['Drug_smile'].unique()
        
        for smile in unique_smiles:
            if smile not in self._smiles_cache:
                encoded, mask = self.smiles_encoder.encode(smile)
                self._smiles_cache[smile] = (encoded, mask)
        
        logger.info(f"Cached {len(self._smiles_cache)} unique SMILES encodings")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample
        
        Returns:
            drug_encoded: Drug SMILES encoding [50]
            se_indices: Side effect substructure indices [50]
            drug_mask: Drug attention mask [50]
            se_mask: Side effect attention mask [50]
            label: Label (0 or 1)
        """
        # Get actual index in dataframe
        data_idx = self.indices[idx]
        
        # Get drug SMILES and side effect ID
        drug_smile = self.df.iloc[data_idx]['Drug_smile']
        se_id = int(self.df.iloc[data_idx]['SE_id'])
        label = self.labels[data_idx]
        
        # Encode drug SMILES
        if self.cache_encoded and drug_smile in self._smiles_cache:
            drug_encoded, drug_mask = self._smiles_cache[drug_smile]
        else:
            drug_encoded, drug_mask = self.smiles_encoder.encode(drug_smile)
            if self.cache_encoded:
                self._smiles_cache[drug_smile] = (drug_encoded, drug_mask)
        
        # Get side effect encoding
        se_indices = self.se_index[se_id, :]
        se_mask_val = self.se_mask[se_id, :]
        
        # Convert to tensors
        drug_encoded = torch.from_numpy(drug_encoded).long()
        se_indices = torch.from_numpy(se_indices).long()
        drug_mask = torch.from_numpy(drug_mask).long()
        se_mask_val = torch.from_numpy(se_mask_val).long()
        label = torch.tensor(int(label), dtype=torch.float32)
        
        return drug_encoded, se_indices, drug_mask, se_mask_val, label
    
    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics"""
        labels = self.labels[self.indices]
        return {
            'total_samples': len(self),
            'positive_samples': int(np.sum(labels > 0)),
            'negative_samples': int(np.sum(labels == 0)),
            'positive_ratio': float(np.mean(labels > 0)),
            'unique_drugs': len(self.df['Drug_smile'].unique()),
            'unique_side_effects': len(self.df['SE_id'].unique())
        }


class DrugSideEffectDataModule:
    """
    Data module for handling data loading and preprocessing
    Optimized for PyTorch 2.x
    """
    
    def __init__(
        self,
        config: Config,
        smiles_encoder,
        fold: int = 0
    ):
        """
        Args:
            config: Configuration object
            smiles_encoder: Function to encode SMILES strings
            fold: Current fold number for cross-validation
        """
        self.config = config
        self.data_config = config.data
        self.dataloader_config = config.dataloader
        self.smiles_encoder = smiles_encoder
        self.fold = fold
        
        # Datasets
        self.train_dataset: Optional[DrugSideEffectDataset] = None
        self.val_dataset: Optional[DrugSideEffectDataset] = None
        self.test_dataset: Optional[DrugSideEffectDataset] = None
    
    def setup(
        self,
        train_df: pd.DataFrame,
        train_indices: np.ndarray,
        train_labels: np.ndarray,
        val_df: pd.DataFrame,
        val_indices: np.ndarray,
        val_labels: np.ndarray,
        se_index: np.ndarray,
        se_mask: np.ndarray
    ):
        """
        Setup datasets with data splits
        
        Args:
            train_df: Training dataframe
            train_indices: Training indices
            train_labels: Training labels
            val_df: Validation dataframe
            val_indices: Validation indices
            val_labels: Validation labels
            se_index: Side effect indices array
            se_mask: Side effect mask array
        """
        logger.info(f"Setting up datasets for fold {self.fold}...")
        
        # Create training dataset
        self.train_dataset = DrugSideEffectDataset(
            df=train_df,
            indices=train_indices,
            labels=train_labels,
            se_index=se_index,
            se_mask=se_mask,
            smiles_encoder=self.smiles_encoder,
            fold=self.fold,
            cache_encoded=True
        )
        
        # Create validation dataset
        self.val_dataset = DrugSideEffectDataset(
            df=val_df,
            indices=val_indices,
            labels=val_labels,
            se_index=se_index,
            se_mask=se_mask,
            smiles_encoder=self.smiles_encoder,
            fold=self.fold,
            cache_encoded=True
        )
        
        # Log statistics
        train_stats = self.train_dataset.get_statistics()
        val_stats = self.val_dataset.get_statistics()
        
        logger.info(f"Train set: {train_stats['total_samples']} samples "
                   f"({train_stats['positive_ratio']:.2%} positive)")
        logger.info(f"Val set: {val_stats['total_samples']} samples "
                   f"({val_stats['positive_ratio']:.2%} positive)")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader optimized for PyTorch 2.x"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.dataloader_config.shuffle_train,
            num_workers=self.dataloader_config.num_workers,
            pin_memory=self.dataloader_config.pin_memory,
            persistent_workers=self.dataloader_config.persistent_workers 
                if self.dataloader_config.num_workers > 0 else False,
            prefetch_factor=self.dataloader_config.prefetch_factor 
                if self.dataloader_config.num_workers > 0 else None,
            drop_last=self.dataloader_config.drop_last_train
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.dataloader_config.shuffle_val,
            num_workers=self.dataloader_config.num_workers,
            pin_memory=self.dataloader_config.pin_memory,
            persistent_workers=self.dataloader_config.persistent_workers 
                if self.dataloader_config.num_workers > 0 else False,
            prefetch_factor=self.dataloader_config.prefetch_factor 
                if self.dataloader_config.num_workers > 0 else None,
            drop_last=self.dataloader_config.drop_last_val
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        if self.test_dataset is None:
            raise ValueError("Test dataset not setup. Call setup() with test data first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.dataloader_config.shuffle_test,
            num_workers=self.dataloader_config.num_workers,
            pin_memory=self.dataloader_config.pin_memory,
            persistent_workers=self.dataloader_config.persistent_workers 
                if self.dataloader_config.num_workers > 0 else False,
            prefetch_factor=self.dataloader_config.prefetch_factor 
                if self.dataloader_config.num_workers > 0 else None,
            drop_last=False
        )


def collate_fn_custom(batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
    """
    Custom collate function for batching
    Can be extended for dynamic padding or other custom logic
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched tensors
    """
    drugs, ses, drug_masks, se_masks, labels = zip(*batch)
    
    return (
        torch.stack(drugs),
        torch.stack(ses),
        torch.stack(drug_masks),
        torch.stack(se_masks),
        torch.stack(labels)
    )


class CachedDataset(Dataset):
    """
    Dataset with full caching for faster epoch iterations
    Useful when dataset fits in memory
    """
    
    def __init__(self, base_dataset: DrugSideEffectDataset):
        """
        Args:
            base_dataset: Base dataset to cache
        """
        self.base_dataset = base_dataset
        self._cache = []
        
        logger.info("Caching entire dataset in memory...")
        for idx in range(len(base_dataset)):
            self._cache.append(base_dataset[idx])
        logger.info(f"Cached {len(self._cache)} samples")
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self._cache[idx]


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
    """
    Convenience function to create train and validation dataloaders
    
    Returns:
        train_loader, val_loader
    """
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
        se_mask=se_mask
    )
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    from config import get_default_config
    
    config = get_default_config()
    
    # Create dummy data for testing
    df = pd.DataFrame({
        'SE_id': np.random.randint(0, 10, 100),
        'Drug_smile': ['CCO'] * 100,
        'Label': np.random.randint(0, 2, 100)
    })
    
    indices = np.arange(100)
    labels = df['Label'].values
    se_index = np.random.randint(0, 100, (994, 50))
    se_mask = np.ones((994, 50))
    
    # Dummy SMILES encoder
    def dummy_encoder(smile):
        return np.random.randint(0, 2586, 50), np.ones(50)
    
    # Create dataset
    dataset = DrugSideEffectDataset(
        df=df,
        indices=indices,
        labels=labels,
        se_index=se_index,
        se_mask=se_mask,
        smiles_encoder=dummy_encoder,
        fold=0,
        cache_encoded=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Statistics: {dataset.get_statistics()}")
    
    # Test dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    
    # Test batch
    batch = next(iter(dataloader))
    drug, se, drug_mask, se_mask, label = batch
    
    print(f"\nBatch shapes:")
    print(f"Drug: {drug.shape}")
    print(f"SE: {se.shape}")
    print(f"Drug mask: {drug_mask.shape}")
    print(f"SE mask: {se_mask.shape}")
    print(f"Label: {label.shape}")
    
    print("\nDataset test passed!")