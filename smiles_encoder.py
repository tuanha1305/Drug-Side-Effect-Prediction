"""
SMILES encoding utilities for drug side effect prediction
Handles conversion of SMILES strings to numerical representations using BPE
"""

import csv
import codecs
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from functools import lru_cache
import logging

from subword_nmt.apply_bpe import BPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMILESEncoder:
    """
    Encoder for SMILES strings using Byte Pair Encoding (BPE)
    """

    def __init__(
            self,
            vocab_path: str,
            subword_map_path: str,
            max_len: int = 50
    ):
        """
        Initialize SMILES encoder

        Args:
            vocab_path: Path to BPE vocabulary file
            subword_map_path: Path to subword mapping CSV
            max_len: Maximum sequence length
        """
        self.max_len = max_len

        # Load BPE codes
        logger.info(f"Loading BPE vocabulary from: {vocab_path}")
        bpe_codes = codecs.open(vocab_path)
        self.bpe = BPE(bpe_codes, merges=-1, separator='')

        # Load subword mapping
        logger.info(f"Loading subword mapping from: {subword_map_path}")
        sub_csv = pd.read_csv(subword_map_path)
        self.idx2word = sub_csv['index'].values
        self.word2idx = dict(zip(self.idx2word, range(len(self.idx2word))))

        self.vocab_size = len(self.word2idx)
        logger.info(f"Vocabulary size: {self.vocab_size}")

    def encode(self, smile: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a SMILES string to indices

        Args:
            smile: SMILES string

        Returns:
            encoded: Encoded indices [max_len]
            mask: Attention mask [max_len] (1 for valid, 0 for padding)
        """
        # Split SMILES using BPE
        tokens = self.bpe.process_line(smile).split()

        try:
            # Convert tokens to indices
            indices = np.array([self.word2idx[token] for token in tokens])
        except KeyError:
            # Handle unknown tokens
            logger.warning(f"Unknown token in SMILES: {smile[:50]}...")
            indices = np.array([0])  # Use padding index

        length = len(indices)

        if length < self.max_len:
            # Pad sequence
            encoded = np.pad(
                indices,
                (0, self.max_len - length),
                'constant',
                constant_values=0
            )
            mask = np.array([1] * length + [0] * (self.max_len - length))
        else:
            # Truncate sequence
            encoded = indices[:self.max_len]
            mask = np.ones(self.max_len)

        return encoded.astype(np.int64), mask.astype(np.int64)

    def encode_batch(
            self,
            smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a batch of SMILES strings

        Args:
            smiles_list: List of SMILES strings

        Returns:
            encoded_batch: [batch_size, max_len]
            mask_batch: [batch_size, max_len]
        """
        encoded_list = []
        mask_list = []

        for smile in smiles_list:
            encoded, mask = self.encode(smile)
            encoded_list.append(encoded)
            mask_list.append(mask)

        return np.stack(encoded_list), np.stack(mask_list)

    def decode(self, indices: np.ndarray) -> str:
        """
        Decode indices back to SMILES string

        Args:
            indices: Encoded indices

        Returns:
            smile: SMILES string
        """
        # Remove padding (index 0)
        indices = indices[indices != 0]

        # Convert indices to tokens
        tokens = [self.idx2word[idx] for idx in indices if idx < len(self.idx2word)]

        # Join tokens
        smile = ''.join(tokens)

        return smile

    def get_statistics(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Get encoding statistics for a list of SMILES

        Args:
            smiles_list: List of SMILES strings

        Returns:
            statistics: Dict with statistics
        """
        lengths = []
        num_unknown = 0

        for smile in smiles_list:
            tokens = self.bpe.process_line(smile).split()
            lengths.append(len(tokens))

            # Check for unknown tokens
            for token in tokens:
                if token not in self.word2idx:
                    num_unknown += 1
                    break

        return {
            'num_samples': len(smiles_list),
            'avg_length': np.mean(lengths),
            'max_length': np.max(lengths),
            'min_length': np.min(lengths),
            'std_length': np.std(lengths),
            'num_unknown': num_unknown,
            'percent_truncated': sum(l > self.max_len for l in lengths) / len(lengths) * 100
        }


class CachedSMILESEncoder:
    """
    SMILES encoder with LRU cache for faster encoding
    """

    def __init__(
            self,
            vocab_path: str,
            subword_map_path: str,
            max_len: int = 50,
            cache_size: int = 10000
    ):
        """
        Initialize cached SMILES encoder

        Args:
            vocab_path: Path to BPE vocabulary file
            subword_map_path: Path to subword mapping CSV
            max_len: Maximum sequence length
            cache_size: LRU cache size
        """
        self.encoder = SMILESEncoder(vocab_path, subword_map_path, max_len)
        self.cache_size = cache_size

        # Create cached encode function
        self._cached_encode = lru_cache(maxsize=cache_size)(self._encode_impl)

    def _encode_impl(self, smile: str) -> Tuple[tuple, tuple]:
        """Internal encode implementation (for caching)"""
        encoded, mask = self.encoder.encode(smile)
        # Convert to tuples for hashability
        return tuple(encoded), tuple(mask)

    def encode(self, smile: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode with caching

        Args:
            smile: SMILES string

        Returns:
            encoded: Encoded indices [max_len]
            mask: Attention mask [max_len]
        """
        encoded_tuple, mask_tuple = self._cached_encode(smile)
        return np.array(encoded_tuple), np.array(mask_tuple)

    def get_cache_info(self) -> Dict:
        """Get cache statistics"""
        info = self._cached_encode.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'size': info.currsize,
            'maxsize': info.maxsize,
            'hit_rate': info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0
        }

    def clear_cache(self):
        """Clear the cache"""
        self._cached_encode.cache_clear()


def load_drug_smiles(
        file_path: str
) -> Tuple[Dict[str, int], List[str]]:
    """
    Load drug SMILES from CSV file

    Args:
        file_path: Path to CSV file with format: drug_name,smiles

    Returns:
        drug_dict: Dictionary mapping drug name to index
        drug_smiles: List of SMILES strings
    """
    logger.info(f"Loading drug SMILES from: {file_path}")

    reader = csv.reader(open(file_path))

    drug_dict = {}
    drug_smiles = []

    for item in reader:
        name = item[0]
        smile = item[1]

        # Avoid duplicates
        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos

        drug_smiles.append(smile)

    logger.info(f"Loaded {len(drug_smiles)} drugs")

    return drug_dict, drug_smiles


def create_smiles_encoder(
        vocab_path: str,
        subword_map_path: str,
        max_len: int = 50,
        use_cache: bool = True,
        cache_size: int = 10000
) -> CachedSMILESEncoder | SMILESEncoder:
    """
    Factory function to create SMILES encoder

    Args:
        vocab_path: Path to BPE vocabulary
        subword_map_path: Path to subword mapping
        max_len: Maximum sequence length
        use_cache: Whether to use cached encoder
        cache_size: Cache size if using cache

    Returns:
        encoder: SMILESEncoder or CachedSMILESEncoder
    """
    if use_cache:
        return CachedSMILESEncoder(
            vocab_path,
            subword_map_path,
            max_len,
            cache_size
        )
    else:
        return SMILESEncoder(
            vocab_path,
            subword_map_path,
            max_len
        )


def validate_smiles_encoding(
        encoder: SMILESEncoder,
        smiles_list: List[str],
        n_samples: int = 5
):
    """
    Validate SMILES encoding by encoding and decoding

    Args:
        encoder: SMILESEncoder instance
        smiles_list: List of SMILES to validate
        n_samples: Number of samples to show
    """
    logger.info("Validating SMILES encoding...")

    for i, smile in enumerate(smiles_list[:n_samples]):
        # Encode
        encoded, mask = encoder.encode(smile)

        # Decode
        decoded = encoder.decode(encoded)

        # Compare
        logger.info(f"\nSample {i + 1}:")
        logger.info(f"  Original:  {smile[:80]}")
        logger.info(f"  Decoded:   {decoded[:80]}")
        logger.info(f"  Encoded shape: {encoded.shape}")
        logger.info(f"  Mask sum: {mask.sum()}/{len(mask)}")
        logger.info(f"  Match: {smile == decoded}")
