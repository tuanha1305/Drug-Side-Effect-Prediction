"""
Inference script for Drug Side Effect Prediction
Based on HSTrans paper architecture

Usage:
    python inference.py --drug_smiles "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O" --se_id 0
    python inference.py --drug_name "ibuprofen" --se_id 0
    python inference.py --interactive
"""

import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pandas as pd

from model import create_model
from config import Config, ModelConfig
from smiles_encoder import create_smiles_encoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DrugSideEffectPredictor:
    """
    Predictor class for drug side effect prediction
    
    Loads model, encoders, and provides prediction interface
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        fold: int = 0
    ):
        """
        Initialize predictor
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            config: Configuration object (if None, uses default)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            fold: Fold number for loading SE substructure data
        """
        self.fold = fold
        
        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Load config
        if config is None:
            self.config = Config()
        else:
            self.config = config
        
        # Load SMILES encoder
        logger.info("Loading SMILES encoder...")
        vocab_path = self.config.data.raw_data_dir / self.config.data.vocab_file
        subword_map_path = self.config.data.raw_data_dir / self.config.data.subword_map_file
        
        self.smiles_encoder = create_smiles_encoder(
            vocab_path=str(vocab_path),
            subword_map_path=str(subword_map_path),
            max_len=self.config.data.max_drug_len,
            use_cache=True
        )
        logger.info(f"✓ SMILES encoder loaded (vocab size: {self.smiles_encoder.encoder.vocab_size})")
        
        # Load side effect data
        logger.info(f"Loading side effect data (fold {fold})...")
        self.se_index, self.se_mask = self._load_side_effect_data(fold)
        logger.info(f"✓ Loaded SE data: {self.se_index.shape}")
        
        # Load drug SMILES database (optional, for lookup by name)
        self.drug_db = self._load_drug_database()
        
        # Load model
        logger.info("Loading model...")
        self.model = self._load_model(checkpoint_path)
        logger.info("✓ Model loaded successfully!")
        
    def _load_side_effect_data(self, fold: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load side effect substructure indices and masks"""
        se_index_path = self.config.data.processed_data_dir / f"SE_sub_index_50_{fold}.npy"
        se_mask_path = self.config.data.processed_data_dir / f"SE_sub_mask_50_{fold}.npy"
        
        if not se_index_path.exists():
            raise FileNotFoundError(f"Side effect index file not found: {se_index_path}")
        if not se_mask_path.exists():
            raise FileNotFoundError(f"Side effect mask file not found: {se_mask_path}")
        
        se_index = np.load(se_index_path)
        se_mask = np.load(se_mask_path)
        
        return se_index, se_mask
    
    def _load_drug_database(self) -> Optional[Dict[str, str]]:
        """Load drug SMILES database for lookup by name"""
        drug_smiles_path = self.config.data.raw_data_dir / self.config.data.drug_smiles_file
        
        if not drug_smiles_path.exists():
            logger.warning(f"Drug SMILES file not found: {drug_smiles_path}")
            return None
        
        try:
            import csv
            drug_db = {}
            with open(drug_smiles_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        name = row[0].lower().strip()
                        smiles = row[1].strip()
                        drug_db[name] = smiles
            logger.info(f"✓ Loaded {len(drug_db)} drugs from database")
            return drug_db
        except Exception as e:
            logger.warning(f"Could not load drug database: {e}")
            return None
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create model
        model = create_model(self.config.model, self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'best_metric' in checkpoint:
                logger.info(f"  Best metric: {checkpoint['best_metric']:.4f}")
        else:
            # Assume checkpoint is state dict directly
            model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.eval()
        
        return model
    
    def predict(
        self,
        drug_smiles: str,
        se_id: int,
        return_prob: bool = False
    ) -> float:
        """
        Predict side effect severity for a drug-side effect pair
        
        Args:
            drug_smiles: SMILES string of the drug
            se_id: Side effect ID (0-993)
            return_prob: If True, return probability; otherwise raw score
        
        Returns:
            prediction: Predicted severity score or probability
        """
        # Validate inputs
        if se_id < 0 or se_id >= self.se_index.shape[0]:
            raise ValueError(f"Invalid SE ID: {se_id}. Must be 0-{self.se_index.shape[0]-1}")
        
        # Encode drug SMILES
        try:
            drug_encoded, drug_mask = self.smiles_encoder.encode(drug_smiles)
        except Exception as e:
            raise ValueError(f"Failed to encode SMILES: {e}")
        
        # Prepare inputs
        drug_tensor = torch.from_numpy(drug_encoded).unsqueeze(0).to(self.device)
        drug_mask_tensor = torch.from_numpy(drug_mask).unsqueeze(0).to(self.device)
        
        se_tensor = torch.from_numpy(self.se_index[se_id]).unsqueeze(0).to(self.device)
        se_mask_tensor = torch.from_numpy(self.se_mask[se_id]).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output, _, _ = self.model(
                drug_tensor,
                se_tensor,
                drug_mask_tensor,
                se_mask_tensor
            )
        
        prediction = output.item()
        
        return prediction
    
    def predict_by_drug_name(
        self,
        drug_name: str,
        se_id: int,
        return_prob: bool = False
    ) -> float:
        """
        Predict side effect severity using drug name
        
        Args:
            drug_name: Name of the drug (e.g., "ibuprofen")
            se_id: Side effect ID
            return_prob: If True, return probability; otherwise raw score
        
        Returns:
            prediction: Predicted severity score or probability
        """
        if self.drug_db is None:
            raise ValueError("Drug database not loaded. Cannot look up drug by name.")
        
        drug_name_lower = drug_name.lower().strip()
        
        if drug_name_lower not in self.drug_db:
            # Try with spaces replaced by dots
            drug_name_alt = drug_name_lower.replace(' ', '.')
            if drug_name_alt in self.drug_db:
                drug_smiles = self.drug_db[drug_name_alt]
            else:
                available = list(self.drug_db.keys())[:10]
                raise ValueError(
                    f"Drug '{drug_name}' not found in database.\n"
                    f"Available drugs (first 10): {available}"
                )
        else:
            drug_smiles = self.drug_db[drug_name_lower]
        
        logger.info(f"Found drug '{drug_name}' with SMILES: {drug_smiles}")
        
        return self.predict(drug_smiles, se_id, return_prob)
    
    def predict_all_side_effects(
        self,
        drug_smiles: str,
        top_k: int = 10,
        return_prob: bool = False
    ) -> List[Tuple[int, float]]:
        """
        Predict all side effects for a drug and return top-k
        
        Args:
            drug_smiles: SMILES string of the drug
            top_k: Number of top predictions to return
            return_prob: If True, return probabilities; otherwise raw scores
        
        Returns:
            predictions: List of (se_id, score) tuples sorted by score (descending)
        """
        logger.info(f"Predicting all side effects for drug...")
        
        predictions = []
        
        # Encode drug once
        drug_encoded, drug_mask = self.smiles_encoder.encode(drug_smiles)
        drug_tensor = torch.from_numpy(drug_encoded).unsqueeze(0).to(self.device)
        drug_mask_tensor = torch.from_numpy(drug_mask).unsqueeze(0).to(self.device)
        
        # Predict for all side effects
        with torch.no_grad():
            for se_id in range(self.se_index.shape[0]):
                se_tensor = torch.from_numpy(self.se_index[se_id]).unsqueeze(0).to(self.device)
                se_mask_tensor = torch.from_numpy(self.se_mask[se_id]).unsqueeze(0).to(self.device)
                
                output, _, _ = self.model(
                    drug_tensor,
                    se_tensor,
                    drug_mask_tensor,
                    se_mask_tensor
                )
                
                score = output.item()
                
                predictions.append((se_id, score))
        
        # Sort by score (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_k]
    
    def predict_batch(
        self,
        drug_smiles_list: List[str],
        se_id_list: List[int],
        return_prob: bool = False
    ) -> np.ndarray:
        """
        Predict for a batch of drug-side effect pairs
        
        Args:
            drug_smiles_list: List of SMILES strings
            se_id_list: List of side effect IDs
            return_prob: If True, return probabilities; otherwise raw scores
        
        Returns:
            predictions: Array of predictions [batch_size]
        """
        if len(drug_smiles_list) != len(se_id_list):
            raise ValueError("drug_smiles_list and se_id_list must have same length")
        
        batch_size = len(drug_smiles_list)
        predictions = np.zeros(batch_size)
        
        # Process in batches
        for i in range(batch_size):
            predictions[i] = self.predict(
                drug_smiles_list[i],
                se_id_list[i],
                return_prob
            )
        
        return predictions


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Drug Side Effect Prediction Inference'
    )
    
    # Input arguments
    parser.add_argument(
        '--drug_smiles',
        type=str,
        help='SMILES string of the drug'
    )
    parser.add_argument(
        '--drug_name',
        type=str,
        help='Name of the drug (e.g., "ibuprofen")'
    )
    parser.add_argument(
        '--se_id',
        type=int,
        default=0,
        help='Side effect ID (0-993)'
    )
    
    # Model arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Fold number for SE data (0-9)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to run on'
    )
    
    # Output arguments
    parser.add_argument(
        '--return_prob',
        action='store_true',
        help='Return probability instead of raw score'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of top predictions to show (for predict_all mode)'
    )
    parser.add_argument(
        '--predict_all',
        action='store_true',
        help='Predict all side effects for the drug'
    )
    
    # Interactive mode
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    return parser.parse_args()


def interactive_mode(predictor: DrugSideEffectPredictor):
    """Run predictor in interactive mode"""
    print("\n" + "="*70)
    print("Drug Side Effect Prediction - Interactive Mode")
    print("="*70)
    print("\nCommands:")
    print("  1. Predict by SMILES: predict <SMILES> <SE_ID>")
    print("  2. Predict by name:   predict_name <DRUG_NAME> <SE_ID>")
    print("  3. Predict all:       predict_all <SMILES>")
    print("  4. Show example:      example")
    print("  5. Quit:              quit")
    print("="*70 + "\n")
    
    # Show example drugs
    if predictor.drug_db:
        print("Available drugs (sample):")
        drugs = list(predictor.drug_db.keys())[:10]
        for drug in drugs:
            print(f"  - {drug}")
        print()
    
    while True:
        try:
            command = input(">>> ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'quit' or cmd == 'exit':
                print("Goodbye!")
                break
            
            elif cmd == 'example':
                # Show Ibuprofen example (as in paper)
                print("\n--- Example: Ibuprofen ---")
                drug_smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
                se_id = 0
                
                print(f"Drug SMILES: {drug_smiles}")
                print(f"Side Effect ID: {se_id}")
                
                score = predictor.predict(drug_smiles, se_id, return_prob=False)
                prob = predictor.predict(drug_smiles, se_id, return_prob=True)
                
                print(f"\nPrediction:")
                print(f"  Raw score: {score:.4f}")
                print(f"  Probability: {prob:.4f}")
                print()
            
            elif cmd == 'predict':
                if len(parts) < 3:
                    print("Usage: predict <SMILES> <SE_ID>")
                    continue
                
                drug_smiles = parts[1]
                se_id = int(parts[2])
                
                score = predictor.predict(drug_smiles, se_id, return_prob=False)
                prob = predictor.predict(drug_smiles, se_id, return_prob=True)
                
                print(f"\nPrediction:")
                print(f"  Drug SMILES: {drug_smiles}")
                print(f"  Side Effect ID: {se_id}")
                print(f"  Raw score: {score:.4f}")
                print(f"  Probability: {prob:.4f}")
                print()
            
            elif cmd == 'predict_name':
                if len(parts) < 3:
                    print("Usage: predict_name <DRUG_NAME> <SE_ID>")
                    continue
                
                drug_name = parts[1]
                se_id = int(parts[2])
                
                score = predictor.predict_by_drug_name(drug_name, se_id, return_prob=False)
                prob = predictor.predict_by_drug_name(drug_name, se_id, return_prob=True)
                
                print(f"\nPrediction:")
                print(f"  Drug: {drug_name}")
                print(f"  Side Effect ID: {se_id}")
                print(f"  Raw score: {score:.4f}")
                print(f"  Probability: {prob:.4f}")
                print()
            
            elif cmd == 'predict_all':
                if len(parts) < 2:
                    print("Usage: predict_all <SMILES>")
                    continue
                
                drug_smiles = parts[1]
                
                print(f"\nPredicting all side effects for: {drug_smiles}")
                predictions = predictor.predict_all_side_effects(
                    drug_smiles,
                    top_k=10,
                    return_prob=True
                )
                
                print("\nTop 10 predicted side effects:")
                for i, (se_id, prob) in enumerate(predictions, 1):
                    print(f"  {i}. SE ID {se_id:3d}: {prob:.4f}")
                print()
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'example' to see an example, or 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            logger.exception(e)


def main():
    """Main function"""
    args = parse_args()
    
    # Setup device
    device = args.device
    if device == 'auto':
        device = None
    
    # Create predictor
    try:
        predictor = DrugSideEffectPredictor(
            checkpoint_path=args.checkpoint,
            device=device,
            fold=args.fold
        )
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode(predictor)
        return
    
    # Single prediction mode
    if args.drug_smiles is None and args.drug_name is None:
        logger.error("Please provide either --drug_smiles or --drug_name")
        print("\nExample usage:")
        print("  python inference.py --drug_smiles 'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O' --se_id 0")
        print("  python inference.py --drug_name ibuprofen --se_id 0")
        print("  python inference.py --interactive")
        return
    
    # Predict all side effects
    if args.predict_all:
        if args.drug_smiles is None:
            logger.error("--predict_all requires --drug_smiles")
            return
        
        print(f"\nPredicting all side effects for: {args.drug_smiles}")
        predictions = predictor.predict_all_side_effects(
            args.drug_smiles,
            top_k=args.top_k,
            return_prob=args.return_prob
        )
        
        print(f"\nTop {args.top_k} predicted side effects:")
        for i, (se_id, score) in enumerate(predictions, 1):
            score_type = "Predicted frequency"
            print(f"  {i}. SE ID {se_id:3d}: {score_type} = {score:.4f}")
        
        return
    
    # Single prediction
    try:
        if args.drug_smiles:
            score = predictor.predict(
                args.drug_smiles,
                args.se_id,
                return_prob=args.return_prob
            )
            print(f"\nDrug SMILES: {args.drug_smiles}")
        else:
            score = predictor.predict_by_drug_name(
                args.drug_name,
                args.se_id,
                return_prob=args.return_prob
            )
            print(f"\nDrug: {args.drug_name}")
        
        score_type = "Probability" if args.return_prob else "Score"
        print(f"Side Effect ID: {args.se_id}")
        print(f"\nPrediction: {score_type} = {score:.4f}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.exception(e)


if __name__ == "__main__":
    main()

