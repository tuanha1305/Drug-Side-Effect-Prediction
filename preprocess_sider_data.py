#!/usr/bin/env python3
"""
Preprocessing Script for SIDER Dataset
=======================================

This script creates drug_side.pkl from raw SIDER data files.

Input Files:
- Supplementary_Data_1.txt: Drug-side effect associations
- drug_SMILES_759.csv: SMILES strings for drugs

Output:
- data/drug_side.pkl: Binary matrix [759 drugs × 994 side effects]

Usage:
    python preprocess_sider_data.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path


def load_sider_data(sider_file):
    """
    Load SIDER drug-side effect associations.

    Args:
        sider_file: Path to Supplementary_Data_1.txt

    Returns:
        DataFrame with columns: GenericName, SideeffectTerm, FrequencyRatingValue
    """
    print(f"📥 Loading {sider_file}...")

    df = pd.read_csv(sider_file, sep='\t')

    print(f"✅ Loaded {len(df)} associations")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Sample rows:")
    print(df.head())

    return df


def load_smiles_data(smiles_file):
    """
    Load drug SMILES strings.

    Args:
        smiles_file: Path to drug_SMILES_759.csv

    Returns:
        DataFrame with columns: Drug_id, Drug_smile
    """
    print(f"\n📥 Loading {smiles_file}...")

    df = pd.read_csv(smiles_file)

    print(f"✅ Loaded {len(df)} drugs with SMILES")
    print(f"   - Columns: {list(df.columns)}")
    print(df.head())

    return df


def create_mappings(sider_df):
    """
    Create drug and side effect ID mappings.

    Args:
        sider_df: DataFrame from load_sider_data

    Returns:
        drug_to_id, se_to_id, id_to_drug, id_to_se
    """
    print("\n🗺️  Creating ID mappings...")

    # Get unique drugs and side effects (sorted for consistency)
    unique_drugs = sorted(sider_df['GenericName'].unique())
    unique_se = sorted(sider_df['SideeffectTerm'].unique())

    # Create mappings
    drug_to_id = {drug: idx for idx, drug in enumerate(unique_drugs)}
    se_to_id = {se: idx for idx, se in enumerate(unique_se)}

    id_to_drug = {idx: drug for drug, idx in drug_to_id.items()}
    id_to_se = {idx: se for se, idx in se_to_id.items()}

    print(f"✅ Created mappings:")
    print(f"   - Drugs: {len(unique_drugs)}")
    print(f"   - Side Effects: {len(unique_se)}")
    print(f"\n   Sample drug mapping:")
    for i, drug in enumerate(list(unique_drugs)[:5]):
        print(f"     '{drug}' → {i}")
    print(f"\n   Sample SE mapping:")
    for i, se in enumerate(list(unique_se)[:5]):
        print(f"     '{se}' → {i}")

    return drug_to_id, se_to_id, id_to_drug, id_to_se


def create_drug_se_matrix(sider_df, drug_to_id, se_to_id, use_frequency=False):
    """
    Create binary drug-side effect matrix.

    Args:
        sider_df: DataFrame from load_sider_data
        drug_to_id: Drug name to ID mapping
        se_to_id: Side effect name to ID mapping
        use_frequency: If True, use FrequencyRatingValue instead of binary

    Returns:
        numpy array of shape [n_drugs, n_side_effects]
    """
    print("\n🔢 Creating drug-SE matrix...")

    n_drugs = len(drug_to_id)
    n_se = len(se_to_id)

    # Initialize matrix with zeros
    drug_se_matrix = np.zeros((n_drugs, n_se))

    # Fill matrix
    skipped = 0
    for idx, row in sider_df.iterrows():
        drug_name = row['GenericName']
        se_name = row['SideeffectTerm']
        freq = row['FrequencyRatingValue']

        # Get IDs
        if drug_name not in drug_to_id:
            skipped += 1
            continue
        if se_name not in se_to_id:
            skipped += 1
            continue

        drug_id = drug_to_id[drug_name]
        se_id = se_to_id[se_name]

        # Set value
        if use_frequency:
            drug_se_matrix[drug_id, se_id] = freq
        else:
            drug_se_matrix[drug_id, se_id] = 1

    # Calculate statistics
    n_positive = np.sum(drug_se_matrix > 0)
    n_zero = np.sum(drug_se_matrix == 0)
    total = n_drugs * n_se

    print(f"✅ Matrix created:")
    print(f"   - Shape: {drug_se_matrix.shape}")
    print(f"   - Positive entries: {n_positive} ({100 * n_positive / total:.2f}%)")
    print(f"   - Zero entries: {n_zero} ({100 * n_zero / total:.2f}%)")
    print(f"   - Total possible: {total}")
    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} entries (missing mappings)")

    return drug_se_matrix


def save_drug_side_pkl(matrix, drug_to_id, se_to_id, id_to_drug, id_to_se,
                       smiles_df, output_path):
    """
    Save processed data to pickle file.

    Args:
        matrix: Drug-SE matrix [n_drugs, n_side_effects]
        drug_to_id: Drug name to ID mapping
        se_to_id: Side effect name to ID mapping
        id_to_drug: ID to drug name mapping
        id_to_se: ID to side effect name mapping
        smiles_df: DataFrame with SMILES data
        output_path: Path to save pickle file
    """
    print(f"\n💾 Saving to {output_path}...")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create data structure
    # Option 1: Just matrix (what main.py seems to expect)
    data = matrix

    # Option 2: Complete structure (more flexible)
    # Uncomment if you want more metadata
    # data = {
    #     'matrix': matrix,
    #     'drug_to_id': drug_to_id,
    #     'se_to_id': se_to_id,
    #     'id_to_drug': id_to_drug,
    #     'id_to_se': id_to_se,
    #     'smiles': smiles_df
    # }

    # Save to pickle
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    # Get file size
    file_size = os.path.getsize(output_path) / 1024 / 1024  # MB

    print(f"✅ Saved successfully!")
    print(f"   - File size: {file_size:.2f} MB")
    print(f"   - Data type: {type(data)}")
    if isinstance(data, np.ndarray):
        print(f"   - Shape: {data.shape}")
        print(f"   - Dtype: {data.dtype}")


def verify_pickle_file(pickle_path):
    """
    Verify the created pickle file.

    Args:
        pickle_path: Path to drug_side.pkl
    """
    print(f"\n🔍 Verifying {pickle_path}...")

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, np.ndarray):
        print(f"✅ Loaded as numpy array")
        print(f"   - Shape: {data.shape}")
        print(f"   - Dtype: {data.dtype}")
        print(f"   - Min value: {data.min()}")
        print(f"   - Max value: {data.max()}")
        print(f"   - Positive entries: {np.sum(data > 0)}")
        print(f"\n   Sample values (first 5x5):")
        print(data[:5, :5])
    elif isinstance(data, dict):
        print(f"✅ Loaded as dictionary")
        print(f"   - Keys: {list(data.keys())}")
        if 'matrix' in data:
            print(f"   - Matrix shape: {data['matrix'].shape}")
    else:
        print(f"⚠️  Unknown format: {type(data)}")


def main():
    """Main preprocessing pipeline."""

    print("=" * 60)
    print("SIDER Data Preprocessing")
    print("=" * 60)

    # Configuration
    sider_file = 'Supplementary_Data_1.txt'
    smiles_file = 'drug_SMILES_759.csv'
    output_file = 'data/drug_side.pkl'

    # Step 1: Load SIDER data
    sider_df = load_sider_data(sider_file)

    # Step 2: Load SMILES data
    smiles_df = load_smiles_data(smiles_file)

    # Step 3: Create mappings
    drug_to_id, se_to_id, id_to_drug, id_to_se = create_mappings(sider_df)

    # Step 4: Create drug-SE matrix
    drug_se_matrix = create_drug_se_matrix(
        sider_df,
        drug_to_id,
        se_to_id,
        use_frequency=True  # Use frequency ratings like original
    )

    # Step 5: Save to pickle
    save_drug_side_pkl(
        drug_se_matrix,
        drug_to_id,
        se_to_id,
        id_to_drug,
        id_to_se,
        smiles_df,
        output_file
    )

    # Step 6: Verify
    verify_pickle_file(output_file)

    print("\n" + "=" * 60)
    print("✅ Preprocessing Complete!")
    print("=" * 60)
    print(f"\nCreated files:")
    print(f"  - {output_file}")
    print(f"\nYou can now run main.py to train the model.")


if __name__ == '__main__':
    main()