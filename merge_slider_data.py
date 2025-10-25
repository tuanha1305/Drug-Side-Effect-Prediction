#!/usr/bin/env python3
"""
Merge SIDER Raw Data for Hugging Face Dataset

This script combines all SIDER raw data files into a single comprehensive dataset
ready for upload to Hugging Face Hub.

Output: sider_complete_dataset.pkl
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("SIDER Dataset Merger for Hugging Face")
print("=" * 80)

# ============================================================================
# STEP 1: Load all raw data
# ============================================================================
print("\n📥 STEP 1: Loading raw data files...")

# Load SIDER associations
print("   Loading Supplementary_Data_1.txt...")
sider_df = pd.read_csv('Supplementary_Data_1.txt', sep='\t')
print(f"   ✅ Loaded {len(sider_df)} drug-side effect associations")
print(f"      - Unique drugs: {sider_df['GenericName'].nunique()}")
print(f"      - Unique side effects: {sider_df['SideeffectTerm'].nunique()}")

# Load SMILES data
print("\n   Loading drug SMILES...")
smiles_df = pd.read_csv('drug_SMILES_759.csv')
print(f"   ✅ Loaded {len(smiles_df)} drugs with SMILES")

# Extract drug names and SMILES
# The CSV format is: drug_name, SMILES_string
drug_names = list(smiles_df.columns[::2])  # Every other column is drug name
smiles_list = []
for i in range(0, len(smiles_df.columns), 2):
    if i < len(smiles_df.columns):
        drug_name = smiles_df.columns[i]
        smiles_string = smiles_df.iloc[0, i + 1] if i + 1 < len(smiles_df.columns) else None
        if smiles_string and not pd.isna(smiles_string):
            smiles_list.append({
                'drug_name': drug_name,
                'smiles': smiles_string
            })

smiles_dict = pd.DataFrame(smiles_list)
print(f"   ✅ Extracted {len(smiles_dict)} drug-SMILES mappings")

# ============================================================================
# STEP 2: Create comprehensive mappings
# ============================================================================
print("\n🗺️  STEP 2: Creating comprehensive mappings...")

# Drug mapping
unique_drugs = sorted(sider_df['GenericName'].unique())
drug_to_id = {drug: idx for idx, drug in enumerate(unique_drugs)}
id_to_drug = {idx: drug for drug, idx in drug_to_id.items()}

# Side effect mapping
unique_se = sorted(sider_df['SideeffectTerm'].unique())
se_to_id = {se: idx for idx, se in enumerate(unique_se)}
id_to_se = {idx: se for se, idx in se_to_id.items()}

print(f"   ✅ Drug mapping: {len(drug_to_id)} drugs")
print(f"   ✅ Side effect mapping: {len(se_to_id)} side effects")

# ============================================================================
# STEP 3: Create drug-SE matrix
# ============================================================================
print("\n🔢 STEP 3: Creating drug-side effect matrix...")

n_drugs = len(unique_drugs)
n_se = len(unique_se)
drug_se_matrix = np.zeros((n_drugs, n_se))

for idx, row in sider_df.iterrows():
    drug_name = row['GenericName']
    se_name = row['SideeffectTerm']
    freq = row['FrequencyRatingValue']

    drug_id = drug_to_id[drug_name]
    se_id = se_to_id[se_name]

    drug_se_matrix[drug_id, se_id] = freq

print(f"   ✅ Matrix shape: {drug_se_matrix.shape}")
print(f"      - Non-zero entries: {np.sum(drug_se_matrix > 0):,}")
print(f"      - Sparsity: {100 * (1 - np.sum(drug_se_matrix > 0) / drug_se_matrix.size):.2f}%")

# ============================================================================
# STEP 4: Merge with SMILES data
# ============================================================================
print("\n🔗 STEP 4: Merging with SMILES data...")

# Create drug info dataframe
drug_info = []
for drug_id, drug_name in id_to_drug.items():
    # Find SMILES
    smiles_match = smiles_dict[smiles_dict['drug_name'] == drug_name]
    smiles = smiles_match.iloc[0]['smiles'] if len(smiles_match) > 0 else None

    # Get side effect statistics
    drug_se_profile = drug_se_matrix[drug_id, :]
    n_side_effects = np.sum(drug_se_profile > 0)
    total_frequency = np.sum(drug_se_profile)

    drug_info.append({
        'drug_id': drug_id,
        'drug_name': drug_name,
        'smiles': smiles,
        'n_side_effects': int(n_side_effects),
        'total_frequency': float(total_frequency),
        'has_smiles': smiles is not None
    })

drug_info_df = pd.DataFrame(drug_info)
print(f"   ✅ Merged drug information:")
print(f"      - Total drugs: {len(drug_info_df)}")
print(f"      - With SMILES: {drug_info_df['has_smiles'].sum()}")
print(f"      - Without SMILES: {len(drug_info_df) - drug_info_df['has_smiles'].sum()}")

# ============================================================================
# STEP 5: Create side effect info
# ============================================================================
print("\n📊 STEP 5: Creating side effect statistics...")

se_info = []
for se_id, se_name in id_to_se.items():
    # Get drug statistics for this SE
    se_profile = drug_se_matrix[:, se_id]
    n_drugs = np.sum(se_profile > 0)
    total_frequency = np.sum(se_profile)
    avg_frequency = total_frequency / n_drugs if n_drugs > 0 else 0

    se_info.append({
        'se_id': se_id,
        'se_name': se_name,
        'n_drugs': int(n_drugs),
        'total_frequency': float(total_frequency),
        'avg_frequency': float(avg_frequency)
    })

se_info_df = pd.DataFrame(se_info)
print(f"   ✅ Created side effect statistics:")
print(f"      - Total side effects: {len(se_info_df)}")
print(
    f"      - Most common: {se_info_df.nlargest(1, 'n_drugs').iloc[0]['se_name']} ({se_info_df.nlargest(1, 'n_drugs').iloc[0]['n_drugs']} drugs)")

# ============================================================================
# STEP 6: Create associations dataframe
# ============================================================================
print("\n🔗 STEP 6: Creating associations dataframe...")

associations = []
for idx, row in sider_df.iterrows():
    drug_name = row['GenericName']
    se_name = row['SideeffectTerm']
    freq = row['FrequencyRatingValue']

    drug_id = drug_to_id[drug_name]
    se_id = se_to_id[se_name]

    associations.append({
        'drug_id': drug_id,
        'drug_name': drug_name,
        'se_id': se_id,
        'se_name': se_name,
        'frequency_rating': int(freq)
    })

associations_df = pd.DataFrame(associations)
print(f"   ✅ Created associations dataframe: {len(associations_df)} rows")

# ============================================================================
# STEP 7: Create metadata
# ============================================================================
print("\n📝 STEP 7: Creating dataset metadata...")

metadata = {
    'dataset_name': 'SIDER - Side Effect Resource',
    'version': '4.1',
    'created_date': datetime.now().isoformat(),
    'description': 'Complete SIDER dataset with drug-side effect associations, SMILES strings, and frequency ratings',
    'source': 'http://sideeffects.embl.de/',
    'statistics': {
        'n_drugs': len(unique_drugs),
        'n_side_effects': len(unique_se),
        'n_associations': len(associations_df),
        'n_drugs_with_smiles': int(drug_info_df['has_smiles'].sum()),
        'matrix_shape': list(drug_se_matrix.shape),
        'matrix_sparsity': float(100 * (1 - np.sum(drug_se_matrix > 0) / drug_se_matrix.size)),
        'frequency_distribution': {
            'rating_1': int(np.sum(drug_se_matrix == 1)),
            'rating_2': int(np.sum(drug_se_matrix == 2)),
            'rating_3': int(np.sum(drug_se_matrix == 3)),
            'rating_4': int(np.sum(drug_se_matrix == 4)),
            'rating_5': int(np.sum(drug_se_matrix == 5))
        }
    },
    'frequency_ratings': {
        '1': 'Placebo frequency',
        '2': 'Rare',
        '3': 'Infrequent',
        '4': 'Frequent',
        '5': 'Very frequent'
    }
}

print(f"   ✅ Created metadata")

# ============================================================================
# STEP 8: Package everything
# ============================================================================
print("\n📦 STEP 8: Packaging complete dataset...")

complete_dataset = {
    # Metadata
    'metadata': metadata,

    # Raw data
    'sider_raw': sider_df,

    # Structured data
    'drug_info': drug_info_df,
    'side_effect_info': se_info_df,
    'associations': associations_df,

    # Matrix format
    'drug_se_matrix': drug_se_matrix,

    # Mappings
    'drug_to_id': drug_to_id,
    'id_to_drug': id_to_drug,
    'se_to_id': se_to_id,
    'id_to_se': id_to_se,
}

print(f"   ✅ Packaged {len(complete_dataset)} components")

# ============================================================================
# STEP 9: Save to pickle
# ============================================================================
print("\n💾 STEP 9: Saving to pickle file...")

output_file = 'sider_complete_dataset.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(complete_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

import os

file_size = os.path.getsize(output_file) / 1024 / 1024
print(f"   ✅ Saved to {output_file}")
print(f"      File size: {file_size:.2f} MB")

# ============================================================================
# STEP 10: Save metadata as JSON
# ============================================================================
print("\n💾 STEP 10: Saving metadata as JSON...")

json_file = 'sider_metadata.json'
with open(json_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✅ Saved metadata to {json_file}")

# ============================================================================
# STEP 11: Create README
# ============================================================================
print("\n📄 STEP 11: Creating README for Hugging Face...")

readme_content = f"""# SIDER - Side Effect Resource Dataset

## Dataset Description

This dataset contains comprehensive drug-side effect association data from the SIDER database (Side Effect Resource).

### Dataset Statistics

- **Number of Drugs:** {metadata['statistics']['n_drugs']:,}
- **Number of Side Effects:** {metadata['statistics']['n_side_effects']:,}
- **Total Associations:** {metadata['statistics']['n_associations']:,}
- **Drugs with SMILES:** {metadata['statistics']['n_drugs_with_smiles']:,}
- **Matrix Shape:** {metadata['statistics']['matrix_shape']}
- **Sparsity:** {metadata['statistics']['matrix_sparsity']:.2f}%

### Frequency Ratings

1. **Placebo frequency** - Side effect occurs at similar rate to placebo
2. **Rare** - Infrequent occurrence
3. **Infrequent** - Occasional occurrence
4. **Frequent** - Common occurrence
5. **Very frequent** - Very common occurrence

### Dataset Structure

The dataset contains:

1. **drug_info** (DataFrame): Information about each drug
   - `drug_id`: Unique drug identifier
   - `drug_name`: Generic drug name
   - `smiles`: SMILES molecular structure
   - `n_side_effects`: Number of side effects
   - `total_frequency`: Total frequency score
   - `has_smiles`: Whether SMILES is available

2. **side_effect_info** (DataFrame): Information about each side effect
   - `se_id`: Unique side effect identifier
   - `se_name`: Side effect term
   - `n_drugs`: Number of drugs causing this SE
   - `total_frequency`: Total frequency score
   - `avg_frequency`: Average frequency per drug

3. **associations** (DataFrame): Drug-side effect pairs
   - `drug_id`: Drug identifier
   - `drug_name`: Drug name
   - `se_id`: Side effect identifier
   - `se_name`: Side effect name
   - `frequency_rating`: Frequency rating (1-5)

4. **drug_se_matrix** (NumPy array): Dense matrix [{metadata['statistics']['matrix_shape'][0]}, {metadata['statistics']['matrix_shape'][1]}]
   - Rows: Drugs
   - Columns: Side effects
   - Values: Frequency ratings (0-5)

5. **Mappings**: ID to name mappings for drugs and side effects

### Usage

```python
import pickle

# Load dataset
with open('sider_complete_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Access components
drug_info = data['drug_info']
side_effects = data['side_effect_info']
associations = data['associations']
matrix = data['drug_se_matrix']

# Get drug by ID
drug_name = data['id_to_drug'][0]
print(f"Drug 0: {{drug_name}}")

# Get side effect by ID
se_name = data['id_to_se'][0]
print(f"SE 0: {{se_name}}")

# Check if drug 5 causes side effect 10
has_se = matrix[5, 10] > 0
print(f"Drug 5 has SE 10: {{has_se}}")
```

### Source

- **Database:** SIDER 4.1
- **URL:** http://sideeffects.embl.de/
- **Version:** 4.1
- **Created:** {metadata['created_date']}

### Citation

If you use this dataset, please cite:

```
Kuhn M, Letunic I, Jensen LJ, Bork P. The SIDER database of drugs and side effects. 
Nucleic Acids Research. 2016;44(D1):D1075-D1079.
```

### License

This dataset is derived from publicly available SIDER data. Please refer to the original SIDER license.

### Contact

For questions or issues, please open an issue on the dataset repository.
"""

readme_file = 'README.md'
with open(readme_file, 'w') as f:
    f.write(readme_content)
print(f"   ✅ Created {readme_file}")

# ============================================================================
# STEP 12: Verification
# ============================================================================
print("\n🔍 STEP 12: Verifying output...")

# Reload and verify
with open(output_file, 'rb') as f:
    loaded_data = pickle.load(f)

checks = [
    ('Metadata exists', 'metadata' in loaded_data),
    ('Drug info exists', 'drug_info' in loaded_data),
    ('SE info exists', 'side_effect_info' in loaded_data),
    ('Associations exist', 'associations' in loaded_data),
    ('Matrix exists', 'drug_se_matrix' in loaded_data),
    ('Mappings exist', all(k in loaded_data for k in ['drug_to_id', 'id_to_drug', 'se_to_id', 'id_to_se'])),
    ('Matrix shape correct', loaded_data['drug_se_matrix'].shape == (759, 994)),
    ('Drug count correct', len(loaded_data['drug_info']) == 759),
    ('SE count correct', len(loaded_data['side_effect_info']) == 994),
    ('Association count correct', len(loaded_data['associations']) == 37441),
]

all_passed = True
for check_name, result in checks:
    status = '✅' if result else '❌'
    print(f"   {status} {check_name}")
    if not result:
        all_passed = False

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ DATASET CREATION COMPLETE!")
print("=" * 80)

print(f"\n📦 Created files:")
print(f"   1. {output_file} ({file_size:.2f} MB) - Complete dataset")
print(f"   2. {json_file} - Metadata (JSON)")
print(f"   3. {readme_file} - README for Hugging Face")

print(f"\n📊 Dataset summary:")
print(f"   ├─ Drugs: {metadata['statistics']['n_drugs']:,}")
print(f"   ├─ Side effects: {metadata['statistics']['n_side_effects']:,}")
print(f"   ├─ Associations: {metadata['statistics']['n_associations']:,}")
print(f"   ├─ Drugs with SMILES: {metadata['statistics']['n_drugs_with_smiles']:,}")
print(f"   └─ Matrix sparsity: {metadata['statistics']['matrix_sparsity']:.2f}%")

print(f"\n🚀 Ready to upload to Hugging Face!")
print(f"   huggingface-cli upload <your-username>/sider-dataset .")

if all_passed:
    print(f"\n✅ All verification checks passed!")
else:
    print(f"\n⚠️  Some verification checks failed. Please review.")

print("\n" + "=" * 80)