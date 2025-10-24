"""
Data preprocessing for drug side effect prediction
Handles data loading, negative sampling, and side effect substructure extraction
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from smiles_encoder import SMILESEncoder, load_drug_smiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_drug_side_matrix(file_path: str) -> np.ndarray:
    """
    Load drug-side effect association matrix
    
    Args:
        file_path: Path to matrix file (.mat or .pkl)
        
    Returns:
        matrix: Drug-side effect matrix [n_drugs, n_side_effects]
    """
    logger.info(f"Loading drug-side effect matrix from: {file_path}")
    
    file_path = Path(file_path)
    
    if file_path.suffix == '.mat':
        from scipy import io
        data = io.loadmat(str(file_path))
        # Assuming the matrix is stored with key 'drug_side' or similar
        # Adjust key name based on your actual .mat file
        for key in data.keys():
            if not key.startswith('__'):
                matrix = data[key]
                break
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            matrix = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Matrix shape: {matrix.shape}")
    logger.info(f"Positive samples: {np.sum(matrix > 0)}")
    logger.info(f"Negative samples: {np.sum(matrix == 0)}")
    
    return matrix


def extract_positive_negative_samples(
    drug_side_matrix: np.ndarray,
    addition_negative_strategy: str = 'all'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract positive and negative samples from drug-side effect matrix
    
    Args:
        drug_side_matrix: [n_drugs, n_side_effects] matrix
        addition_negative_strategy: 'all' or number of additional negatives
        
    Returns:
        addition_negative_sample: Additional negative samples
        final_positive_sample: Final positive samples [n, 3] (drug_id, se_id, label)
        final_negative_sample: Final negative samples [n, 3]
    """
    logger.info("Extracting positive and negative samples...")
    
    n_drugs, n_side_effects = drug_side_matrix.shape
    
    # Create interaction target array
    interaction_target = np.zeros((n_drugs * n_side_effects, 3), dtype=int)
    k = 0
    
    for i in range(n_drugs):
        for j in range(n_side_effects):
            interaction_target[k, 0] = i  # drug id
            interaction_target[k, 1] = j  # side effect id
            interaction_target[k, 2] = drug_side_matrix[i, j]  # label
            k += 1
    
    # Sort by label
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    
    # Split positive and negative
    num_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    final_positive_sample = data_shuffle[-num_positive:]
    negative_sample = data_shuffle[:-num_positive]
    
    # Sample negative examples
    num_negative_samples = len(negative_sample)
    indices = list(range(num_negative_samples))
    
    if addition_negative_strategy == 'all':
        sampled_indices = np.random.choice(
            indices, 
            size=num_negative_samples, 
            replace=False
        )
    else:
        n_additional = int(addition_negative_strategy)
        sampled_indices = np.random.choice(
            indices,
            size=(1 + n_additional) * num_positive,
            replace=False
        )
    
    # Split into training negatives and additional negatives
    final_negative_sample = negative_sample[sampled_indices[:num_positive]]
    addition_negative_sample = negative_sample[sampled_indices[num_positive:]]
    
    # Combine positives and negatives for balanced training
    final_sample = np.vstack((final_positive_sample, final_negative_sample))
    
    logger.info(f"Positive samples: {len(final_positive_sample)}")
    logger.info(f"Negative samples (training): {len(final_negative_sample)}")
    logger.info(f"Additional negative samples: {len(addition_negative_sample)}")
    
    return addition_negative_sample, final_positive_sample, final_negative_sample


def identify_side_effect_substructures(
    data: List[Tuple],
    smiles_encoder: SMILESEncoder,
    percentile_threshold: float = 95.0,
    top_k: int = 50,
    fold: int = 0,
    output_dir: Path = Path("data/processed")
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify important substructures for each side effect
    
    Args:
        data: List of (se_id, drug_smile, label) tuples
        smiles_encoder: SMILES encoder instance
        percentile_threshold: Percentile for filtering substructures
        top_k: Number of top substructures to keep
        fold: Fold number
        output_dir: Output directory
        
    Returns:
        se_sub_index: [n_side_effects, top_k] substructure indices
        se_sub_mask: [n_side_effects, top_k] substructure masks
    """
    logger.info(f"Identifying side effect substructures for fold {fold}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    side_ids = [item[0] for item in data]
    drug_smiles = [item[1] for item in data]
    labels = [float(item[2]) for item in data]
    
    n_side_effects = 994
    vocab_size = smiles_encoder.vocab_size
    
    # Get SMILES encodings
    logger.info("Encoding SMILES...")
    sub_dict = {}
    for i in tqdm(range(len(drug_smiles)), desc="Encoding"):
        drug_sub, _ = smiles_encoder.encode(drug_smiles[i])
        sub_dict[i] = drug_sub.tolist()
    
    # Build side effect-substructure matrix
    logger.info("Building SE-substructure matrix...")
    se_sub = np.zeros((n_side_effects, vocab_size))
    
    for j in tqdm(range(len(drug_smiles)), desc="Building matrix"):
        side_id = int(side_ids[j])
        label = labels[j]
        
        for substructure_idx in sub_dict[j]:
            if substructure_idx == 0:  # Skip padding
                continue
            se_sub[side_id][substructure_idx] += label
    
    # Calculate frequencies using statistical measure
    logger.info("Calculating substructure frequencies...")
    
    # Total sum
    n = np.sum(se_sub)
    
    # Row sums (side effect frequencies)
    se_sum = np.sum(se_sub, axis=1)
    se_p = se_sum / n
    
    # Column sums (substructure frequencies)
    sub_sum = np.sum(se_sub, axis=0)
    sub_p = sub_sum / n
    
    # Normalized frequency matrix
    se_sub_p = se_sub / n
    
    # Calculate statistical significance (like chi-square)
    freq = np.zeros((n_side_effects, vocab_size))
    for i in tqdm(range(n_side_effects), desc="Computing frequencies"):
        for j in range(vocab_size):
            if se_p[i] * sub_p[j] == 0:
                freq[i][j] = 0
            else:
                expected = se_p[i] * sub_p[j]
                observed = se_sub_p[i][j]
                
                # Statistical measure
                numerator = (observed - expected)
                denominator = np.sqrt((expected / n) * (1 - se_p[i]) * (1 - sub_p[j]))
                
                if denominator > 0:
                    freq[i][j] = (numerator / denominator) + 1e-5
                else:
                    freq[i][j] = 0
    
    # Save frequency matrix
    freq_path = output_dir / f"freq_{fold}.npy"
    np.save(freq_path, freq)
    logger.info(f"Saved frequency matrix to {freq_path}")
    
    # Filter top-k substructures per side effect
    logger.info(f"Filtering top-{top_k} substructures per side effect...")
    
    # Calculate percentile threshold
    non_nan_values = freq[~np.isnan(freq)]
    if len(non_nan_values) > 0:
        percentile_val = np.percentile(non_nan_values, percentile_threshold)
        logger.info(f"{percentile_threshold}% percentile: {percentile_val:.4f}")
    else:
        percentile_val = 0
    
    # Extract top-k for each side effect
    se_sub_index = np.zeros((n_side_effects, top_k), dtype=int)
    lengths = []
    
    for i in range(n_side_effects):
        # Sort by frequency (descending)
        sorted_indices = np.argsort(freq[i])[::-1]
        
        # Filter by threshold
        filtered_indices = sorted_indices[freq[i][sorted_indices] > percentile_val]
        
        lengths.append(len(filtered_indices))
        
        # Take top-k
        k = 0
        for idx in filtered_indices:
            if k < top_k:
                se_sub_index[i][k] = idx
                k += 1
            else:
                break
    
    logger.info(f"Average substructures per SE: {np.mean(lengths):.2f}")
    logger.info(f"Max substructures: {np.max(lengths)}")
    logger.info(f"Min substructures: {np.min(lengths)}")
    
    # Create mask (1 for valid substructures, 0 for padding)
    se_sub_mask = (se_sub_index > 0).astype(int)
    
    # Save results
    index_path = output_dir / f"SE_sub_index_{top_k}_{fold}.npy"
    mask_path = output_dir / f"SE_sub_mask_{top_k}_{fold}.npy"
    
    np.save(index_path, se_sub_index)
    np.save(mask_path, se_sub_mask)
    
    logger.info(f"Saved SE substructure index to {index_path}")
    logger.info(f"Saved SE substructure mask to {mask_path}")
    
    return se_sub_index, se_sub_mask


def prepare_dataframes(
    final_sample: np.ndarray,
    drug_smiles: List[str]
) -> pd.DataFrame:
    """
    Prepare dataframe for training
    
    Args:
        final_sample: [n, 3] array with (drug_id, se_id, label)
        drug_smiles: List of SMILES strings
        
    Returns:
        df: DataFrame with columns [SE_id, Drug_smile, Label]
    """
    data = []
    
    for i in range(final_sample.shape[0]):
        drug_id = int(final_sample[i, 0])
        se_id = int(final_sample[i, 1])
        label = int(final_sample[i, 2])
        
        data.append({
            'SE_id': se_id,
            'Drug_smile': drug_smiles[drug_id],
            'Label': label
        })
    
    df = pd.DataFrame(data)
    
    logger.info(f"Created dataframe with {len(df)} samples")
    logger.info(f"Positive: {(df['Label'] > 0).sum()}")
    logger.info(f"Negative: {(df['Label'] == 0).sum()}")
    
    return df


def create_cross_validation_splits(
    final_sample: np.ndarray,
    n_folds: int = 10,
    random_state: int = 1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified K-fold splits
    
    Args:
        final_sample: [n, 3] array with (drug_id, se_id, label)
        n_folds: Number of folds
        random_state: Random seed
        
    Returns:
        splits: List of (train_indices, test_indices) tuples
    """
    logger.info(f"Creating {n_folds}-fold cross-validation splits...")
    
    # Extract features and labels
    X = final_sample[:, :2]  # drug_id, se_id
    y = final_sample[:, 2]   # label
    
    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    
    splits = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        splits.append((train_idx, test_idx))
        logger.info(f"Fold {fold}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    return splits


class DataPreprocessor:
    """
    Complete data preprocessing pipeline
    """
    
    def __init__(
        self,
        drug_smiles_path: str,
        drug_side_matrix_path: str,
        vocab_path: str,
        subword_map_path: str,
        output_dir: str = "data/processed"
    ):
        """
        Initialize preprocessor
        
        Args:
            drug_smiles_path: Path to drug SMILES CSV
            drug_side_matrix_path: Path to drug-side effect matrix
            vocab_path: Path to BPE vocabulary
            subword_map_path: Path to subword mapping
            output_dir: Output directory
        """
        self.drug_smiles_path = drug_smiles_path
        self.drug_side_matrix_path = drug_side_matrix_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info("Loading drug SMILES...")
        self.drug_dict, self.drug_smiles = load_drug_smiles(drug_smiles_path)
        
        logger.info("Loading drug-side effect matrix...")
        self.drug_side_matrix = load_drug_side_matrix(drug_side_matrix_path)
        
        # Create SMILES encoder
        logger.info("Creating SMILES encoder...")
        from smiles_encoder import create_smiles_encoder
        self.smiles_encoder = create_smiles_encoder(
            vocab_path,
            subword_map_path,
            max_len=50,
            use_cache=True
        )
    
    def preprocess(
        self,
        n_folds: int = 10,
        random_state: int = 1,
        addition_negative_strategy: str = 'all',
        percentile_threshold: float = 95.0,
        top_k: int = 50
    ):
        """
        Run complete preprocessing pipeline
        
        Args:
            n_folds: Number of cross-validation folds
            random_state: Random seed
            addition_negative_strategy: Negative sampling strategy
            percentile_threshold: Percentile for filtering substructures
            top_k: Number of top substructures
        """
        # Extract samples
        logger.info("\n" + "="*60)
        logger.info("Step 1: Extracting positive and negative samples")
        logger.info("="*60)
        
        addition_neg, final_pos, final_neg = extract_positive_negative_samples(
            self.drug_side_matrix,
            addition_negative_strategy
        )
        
        # Combine samples
        final_sample = np.vstack((final_pos, final_neg))
        
        # Create cross-validation splits
        logger.info("\n" + "="*60)
        logger.info("Step 2: Creating cross-validation splits")
        logger.info("="*60)
        
        splits = create_cross_validation_splits(
            final_sample,
            n_folds,
            random_state
        )
        
        # Prepare dataframe
        logger.info("\n" + "="*60)
        logger.info("Step 3: Preparing dataframe")
        logger.info("="*60)
        
        df = prepare_dataframes(final_sample, self.drug_smiles)
        
        # Save dataframe
        df_path = self.output_dir / "processed_data.csv"
        df.to_csv(df_path, index=False)
        logger.info(f"Saved dataframe to {df_path}")
        
        # Process each fold
        logger.info("\n" + "="*60)
        logger.info("Step 4: Identifying side effect substructures")
        logger.info("="*60)
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"\nProcessing fold {fold}...")
            
            # Get training data for this fold
            train_data = []
            for idx in train_idx:
                row = df.iloc[idx]
                train_data.append((
                    int(row['SE_id']),
                    row['Drug_smile'],
                    int(row['Label'])
                ))
            
            # Identify substructures
            se_sub_index, se_sub_mask = identify_side_effect_substructures(
                data=train_data,
                smiles_encoder=self.smiles_encoder.encoder if hasattr(self.smiles_encoder, 'encoder') else self.smiles_encoder,
                percentile_threshold=percentile_threshold,
                top_k=top_k,
                fold=fold,
                output_dir=self.output_dir
            )
        
        # Save splits
        splits_path = self.output_dir / "cv_splits.pkl"
        with open(splits_path, 'wb') as f:
            pickle.dump(splits, f)
        logger.info(f"\nSaved CV splits to {splits_path}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ Preprocessing completed successfully!")
        logger.info("="*60)


def load_preprocessing_assets(data_dir: str = 'data/processed') -> dict:
    """
    Load all preprocessing assets needed for inference

    Args:
        data_dir: Directory containing processed data

    Returns:
        dict with keys:
            - vocab_file: Path to vocabulary file
            - subword_map_file: Path to subword map file
            - se_index: Side effect index array [num_se, seq_len]
            - se_mask: Side effect mask array [num_se, seq_len]
            - df_data: Main dataframe (optional)
    """
    import pickle
    from pathlib import Path

    data_dir = Path(data_dir)

    # Load drug_side.pkl
    drug_side_path = data_dir / 'drug_side.pkl'
    if not drug_side_path.exists():
        # Try parent directory
        drug_side_path = data_dir.parent / 'raw' / 'drug_side.pkl'

    with open(drug_side_path, 'rb') as f:
        data = pickle.load(f)

    assets = {
        'se_index': data['se_index'],
        'se_mask': data['se_mask'],
        'df_data': data.get('df_data', None)
    }

    # Vocab and subword map files
    vocab_file = data_dir.parent / 'raw' / 'drug_codes_chembl_freq_1500.txt'
    subword_map_file = data_dir.parent / 'raw' / 'subword_units_map_chembl_freq_1500.csv'

    if not vocab_file.exists():
        vocab_file = Path('data/raw/drug_codes_chembl_freq_1500.txt')
    if not subword_map_file.exists():
        subword_map_file = Path('data/raw/subword_units_map_chembl_freq_1500.csv')

    assets['vocab_file'] = str(vocab_file)
    assets['subword_map_file'] = str(subword_map_file)

    return assets

if __name__ == "__main__":
    # Test preprocessing
    print("="*60)
    print("Testing Data Preprocessing")
    print("="*60)
    
    # Paths
    drug_smiles_path = "data/raw/drug_SMILES_750.csv"
    drug_side_matrix_path = "data/raw/drug_side.pkl"
    vocab_path = "data/raw/drug_codes_chembl_freq_1500.txt"
    subword_map_path = "data/raw/subword_units_map_chembl_freq_1500.csv"
    
    # Check if files exist
    if not Path(drug_smiles_path).exists():
        print(f"ERROR: {drug_smiles_path} not found!")
        print("Creating dummy data for testing...")
        
        # Create dummy data
        drug_side_matrix = np.random.randint(0, 2, (750, 994))
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open("data/raw/drug_side.pkl", 'wb') as f:
            pickle.dump(drug_side_matrix, f)
        
        drug_side_matrix_path = "data/raw/drug_side.pkl"
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        drug_smiles_path=drug_smiles_path,
        drug_side_matrix_path=drug_side_matrix_path,
        vocab_path=vocab_path,
        subword_map_path=subword_map_path,
        output_dir="data/processed"
    )
    
    # Run preprocessing (only first fold for testing)
    print("\nRunning preprocessing (1 fold only for testing)...")
    preprocessor.preprocess(
        n_folds=2,  # Only 2 folds for quick testing
        random_state=1,
        addition_negative_strategy='all',
        percentile_threshold=95.0,
        top_k=50
    )
    
    print("\n✓ Preprocessing test completed!")