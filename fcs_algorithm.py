import re
import collections
from tqdm import tqdm
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FCSAlgorithm:
    """
    Implementation of the FCS (Frequent Consecutive Subsequence) algorithm 
    as described in the HSTrans paper (Section 3.1).
    
    Process:
    1. Initialize set V with atomic tokens (characters).
    2. Tokenize corpus into set W.
    3. Iteratively find most frequent pair (A, B) in W.
    4. Merge (A, B) -> AB and add to V.
    5. Repeat until vocab size reached or threshold met.
    """

    def __init__(self, num_merges: int = 1500, min_frequency: int = 2):
        self.num_merges = num_merges
        self.min_frequency = min_frequency
        self.vocab: Set[str] = set()
        self.merges: Dict[Tuple[str, str], str] = {}
        
        # Regex to split SMILES into initial atoms/chars
        # This regex keeps bracketed items [NH] and 2-char atoms Cl, Br together, 
        # everything else is single char.
        self.tokenizer_pattern = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")

    def get_vocab(self, smiles_list: List[str]) -> Dict[Tuple[str, ...], int]:
        """
        Initial tokenization of the corpus (Set W in paper).
        Returns a dictionary of {tokenized_sequence_tuple: frequency}
        """
        vocab = collections.defaultdict(int)
        for smile in tqdm(smiles_list, desc="Initial Tokenization"):
            # Split SMILES into basic tokens
            tokens = self.tokenizer_pattern.findall(smile)
            # Add " " as end of token marker to prevent cross-word merges if needed
            # HSTrans typically treats SMILES as isolated, so tuple is fine
            vocab[tuple(tokens)] += 1
            
            # Add initial tokens to Set V
            for token in tokens:
                self.vocab.add(token)
                
        return vocab

    def get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """
        Scan set W to identify combinations (pairs).
        """
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair: Tuple[str, str], v_in: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """
        Merge pair (A, B) into single entity AB in the corpus.
        """
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        # Since we store vocab as tuples of strings, we iterate and merge manually
        # Optimization: Only iterate words containing both parts
        first, second = pair
        new_token = first + second
        
        for word, freq in v_in.items():
            if first not in word or second not in word:
                v_out[word] = freq
                continue
            
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            v_out[tuple(new_word)] = freq
            
        return v_out

    def train(self, smiles_list: List[str], output_dir: str):
        """
        Execute the FCS training loop.
        """
        logger.info("Starting FCS Training...")
        
        # 1. Initialize W (vocab with frequencies)
        vocab = self.get_vocab(smiles_list)
        
        # 2. Iterative Merging
        for i in tqdm(range(self.num_merges), desc="FCS Merging"):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
                
            # Identify most frequent combination (A, B)
            best = max(pairs, key=pairs.get)
            
            if pairs[best] < self.min_frequency:
                logger.info(f"Stopping early: Max frequency {pairs[best]} < threshold")
                break
                
            # Merge A and B -> AB
            vocab = self.merge_vocab(best, vocab)
            
            # Add to set V and record merge rule
            self.merges[best] = best[0] + best[1]
            self.vocab.add(best[0] + best[1])
            
        self.save_model(output_dir)
        logger.info("FCS Training Complete.")

    def save_model(self, output_dir: str):
        """
        Save the learned substructures and vocabulary in a format 
        compatible with the provided SMILESEncoder (which uses subword_nmt).
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save BPE codes (This mimics the 'drug_codes.txt' file)
        # Format: token1 token2
        codes_path = out_path / f"drug_codes_chembl_freq_{self.num_merges}.txt"
        with open(codes_path, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n") # Standard BPE header
            for pair in self.merges.keys():
                f.write(f"{pair[0]} {pair[1]}\n")
        
        # 2. Save Subword Map (This mimics 'subword_units_map.csv')
        # Format: index, substructure
        # We need to collect all unique tokens that appear in the final vocabulary
        # plus the base characters.
        
        final_tokens = set()
        # Add single chars
        for token in self.vocab:
            final_tokens.add(token)
            
        # Also ensure special tokens are present
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        
        sorted_tokens = sorted(list(final_tokens))
        all_vocab = special_tokens + sorted_tokens
        
        map_path = out_path / f"subword_units_map_chembl_freq_{self.num_merges}.csv"
        df_map = pd.DataFrame({
            'index': all_vocab,
            # For simple map, the value can just be the token itself or frequency
            # The SMILESEncoder expects 'index' column to be the word
            'freq': [0] * len(all_vocab) # Dummy freq
        })
        df_map.to_csv(map_path, index=False)
        
        logger.info(f"Saved BPE codes to {codes_path}")
        logger.info(f"Saved subword map to {map_path}")

if __name__ == "__main__":
    # Example usage
    # You would typically load your SMILES list from your csv here
    
    # Dummy data for demonstration
    dummy_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O", # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" # Ibuprofen
    ] * 100
    
    # Or load from file if available
    try:
        df = pd.read_csv("data/raw/drug_SMILES_750.csv")
        if 'smiles' in df.columns:
            smiles_list = df['smiles'].tolist()
        elif 'Drug_smile' in df.columns:
            smiles_list = df['Drug_smile'].tolist()
        else:
            smiles_list = df.iloc[:, 1].tolist() # Assume 2nd col is smiles
        logger.info(f"Loaded {len(smiles_list)} SMILES from file")
    except:
        logger.warning("Could not load file, using dummy data")
        smiles_list = dummy_smiles

    fcs = FCSAlgorithm(num_merges=500) # Adjust num_merges as per paper (e.g., 1500)
    fcs.train(smiles_list, output_dir="data/raw_fcs_generated")