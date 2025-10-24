"""
Test script to load real data from project files
"""

import sys
import numpy as np
import pandas as pd
import pickle
import codecs
from pathlib import Path
import torch
from subword_nmt.apply_bpe import BPE

# Import our modules
from config import get_default_config
from dataset import DrugSideEffectDataset, create_dataloaders


def load_drug_smile(file_path):
    """
    Load drug SMILES from CSV file
    Returns:
        drug_dict: dict mapping drug name to index
        drug_smile: list of SMILES strings
    """
    import csv

    print(f"Loading drug SMILES from: {file_path}")
    reader = csv.reader(open(file_path))

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]
        smile = item[1]

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)

    print(f"Loaded {len(drug_smile)} drugs")
    return drug_dict, drug_smile


def create_smiles_encoder(vocab_path, subword_map_path):
    """
    Create SMILES encoder function using BPE
    """
    print(f"Loading vocabulary from: {vocab_path}")
    print(f"Loading subword map from: {subword_map_path}")

    # Load BPE codes
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

    # Load subword mapping
    sub_csv = pd.read_csv(subword_map_path)
    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    print(f"Vocabulary size: {len(words2idx_d)}")

    def drug2emb_encoder(smile, max_len=50):
        """
        Encode SMILES string to indices

        Args:
            smile: SMILES string
            max_len: maximum sequence length

        Returns:
            encoded: numpy array of indices [max_len]
            mask: attention mask [max_len]
        """
        # Split SMILES using BPE
        t1 = dbpe.process_line(smile).split()

        try:
            # Convert substructures to indices
            i1 = np.asarray([words2idx_d[i] for i in t1])
        except:
            # If unknown token, use index 0
            i1 = np.array([0])

        l = len(i1)

        if l < max_len:
            # Pad sequence
            encoded = np.pad(i1, (0, max_len - l), 'constant', constant_values=0)
            mask = ([1] * l) + ([0] * (max_len - l))
        else:
            # Truncate sequence
            encoded = i1[:max_len]
            mask = [1] * max_len

        return encoded, np.asarray(mask)

    return drug2emb_encoder


def load_side_effect_data(fold=0):
    """
    Load pre-computed side effect substructure data
    """
    se_index_path = f"data/processed/SE_sub_index_50_{fold}.npy"
    se_mask_path = f"data/processed/SE_sub_mask_50_{fold}.npy"

    print(f"\nLooking for side effect data:")
    print(f"  - {se_index_path}")
    print(f"  - {se_mask_path}")

    if Path(se_index_path).exists() and Path(se_mask_path).exists():
        print("Loading existing side effect data...")
        se_index = np.load(se_index_path)
        se_mask = np.load(se_mask_path)
    else:
        print("Side effect data not found! Creating dummy data...")
        print("NOTE: You need to run preprocessing first to generate real data")
        se_index = np.random.randint(0, 2586, (994, 50))
        se_mask = np.ones((994, 50))

    print(f"Side effect index shape: {se_index.shape}")
    print(f"Side effect mask shape: {se_mask.shape}")

    return se_index, se_mask


def test_load_data():
    """
    Main test function to load real data
    """
    print("=" * 60)
    print("Testing Data Loading with Real Files")
    print("=" * 60)

    # Get config
    config = get_default_config()
    config.dataloader.num_workers = 0  # For testing

    # Check if data files exist
    drug_smiles_path = Path("data/raw/drug_SMILES_750.csv")
    vocab_path = Path("data/raw/drug_codes_chembl_freq_1500.txt")
    subword_map_path = Path("data/raw/subword_units_map_chembl_freq_1500.csv")

    if not drug_smiles_path.exists():
        print(f"ERROR: {drug_smiles_path} not found!")
        return

    if not vocab_path.exists():
        print(f"ERROR: {vocab_path} not found!")
        return

    if not subword_map_path.exists():
        print(f"ERROR: {subword_map_path} not found!")
        return

    # Load drug SMILES
    print("\n" + "=" * 60)
    print("1. Loading Drug SMILES")
    print("=" * 60)
    drug_dict, drug_smile = load_drug_smile(drug_smiles_path)

    # Create SMILES encoder
    print("\n" + "=" * 60)
    print("2. Creating SMILES Encoder")
    print("=" * 60)
    smiles_encoder = create_smiles_encoder(vocab_path, subword_map_path)

    # Test encoding a few SMILES
    print("\nTesting SMILES encoding:")
    for i in range(min(3, len(drug_smile))):
        smile = drug_smile[i]
        encoded, mask = smiles_encoder(smile)
        print(f"\nSMILES {i}: {smile[:50]}...")
        print(f"  Encoded shape: {encoded.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Encoded (first 10): {encoded[:10]}")
        print(f"  Mask (first 10): {mask[:10]}")

    # Load side effect data
    print("\n" + "=" * 60)
    print("3. Loading Side Effect Data")
    print("=" * 60)
    se_index, se_mask = load_side_effect_data(fold=0)

    # Create dummy dataset for testing
    print("\n" + "=" * 60)
    print("4. Creating Test Dataset")
    print("=" * 60)

    # Create a sample dataframe with drug-side effect pairs
    n_samples = 1000
    df_data = []

    for i in range(n_samples):
        drug_idx = i % len(drug_smile)
        se_id = np.random.randint(0, 994)
        label = np.random.randint(0, 2)
        df_data.append({
            'SE_id': se_id,
            'Drug_smile': drug_smile[drug_idx],
            'Label': label
        })

    df = pd.DataFrame(df_data)
    print(f"Created dataframe with {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Split into train/val
    train_size = int(0.8 * len(df))
    train_indices = np.arange(train_size)
    val_indices = np.arange(train_size, len(df))
    train_labels = df.iloc[train_indices]['Label'].values
    val_labels = df.iloc[val_indices]['Label'].values

    print(f"\nTrain samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")

    # Create dataset
    print("\n" + "=" * 60)
    print("5. Creating PyTorch Dataset")
    print("=" * 60)

    dataset = DrugSideEffectDataset(
        df=df,
        indices=train_indices,
        labels=train_labels,
        se_index=se_index,
        se_mask=se_mask,
        smiles_encoder=smiles_encoder,
        fold=0,
        cache_encoded=True
    )

    print(f"Dataset created successfully!")
    print(f"Dataset size: {len(dataset)}")
    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test getting a sample
    print("\n" + "=" * 60)
    print("6. Testing Data Retrieval")
    print("=" * 60)

    sample = dataset[0]
    drug, se, drug_mask, se_mask_val, label = sample

    print(f"Sample 0:")
    print(f"  Drug shape: {drug.shape}, dtype: {drug.dtype}")
    print(f"  SE shape: {se.shape}, dtype: {se.dtype}")
    print(f"  Drug mask shape: {drug_mask.shape}, dtype: {drug_mask.dtype}")
    print(f"  SE mask shape: {se_mask_val.shape}, dtype: {se_mask_val.dtype}")
    print(f"  Label: {label.item()}, dtype: {label.dtype}")

    # Create DataLoader
    print("\n" + "=" * 60)
    print("7. Testing DataLoader")
    print("=" * 60)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    print(f"DataLoader created with batch_size=32")
    print(f"Number of batches: {len(dataloader)}")

    # Get a batch
    batch = next(iter(dataloader))
    drug_batch, se_batch, drug_mask_batch, se_mask_batch, label_batch = batch

    print(f"\nBatch shapes:")
    print(f"  Drug: {drug_batch.shape}")
    print(f"  SE: {se_batch.shape}")
    print(f"  Drug mask: {drug_mask_batch.shape}")
    print(f"  SE mask: {se_mask_batch.shape}")
    print(f"  Label: {label_batch.shape}")

    print(f"\nBatch dtypes:")
    print(f"  Drug: {drug_batch.dtype}")
    print(f"  SE: {se_batch.dtype}")
    print(f"  Drug mask: {drug_mask_batch.dtype}")
    print(f"  SE mask: {se_mask_batch.dtype}")
    print(f"  Label: {label_batch.dtype}")

    # Test a few batches
    print("\n" + "=" * 60)
    print("8. Testing Multiple Batches")
    print("=" * 60)

    n_test_batches = 3
    for i, batch in enumerate(dataloader):
        if i >= n_test_batches:
            break
        print(f"Batch {i + 1}: OK")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)
    print("\nYou can now use this data pipeline for training!")


if __name__ == "__main__":
    try:
        test_load_data()
    except Exception as e:
        print(f"\n{'=' * 60}")
        print("ERROR occurred during testing:")
        print(f"{'=' * 60}")
        import traceback

        traceback.print_exc()
        print(f"\n{'=' * 60}")