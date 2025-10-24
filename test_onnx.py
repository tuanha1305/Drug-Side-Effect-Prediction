#!/usr/bin/env python3
"""
Test ONNX model inference
Compare with PyTorch model and benchmark performance
"""

import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

import onnxruntime as ort
import torch

from preprocessing import load_preprocessing_assets
from smiles_encoder import CachedSMILESEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXInference:
    """ONNX model inference wrapper"""

    def __init__(
            self,
            onnx_path: str,
            providers: List[str] = None
    ):
        """
        Args:
            onnx_path: Path to ONNX model
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if providers is None:
            providers = ['CPUExecutionProvider']

        logger.info(f"Loading ONNX model from {onnx_path}")
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info(f"✓ Model loaded with providers: {self.session.get_providers()}")

    def predict(
            self,
            drug_encoded: np.ndarray,
            se_indices: np.ndarray,
            drug_mask: np.ndarray,
            se_mask: np.ndarray
    ) -> np.ndarray:
        """
        Run inference

        Args:
            drug_encoded: Drug SMILES encoding [batch, seq_len]
            se_indices: Side effect indices [batch, seq_len]
            drug_mask: Drug mask [batch, seq_len]
            se_mask: Side effect mask [batch, seq_len]

        Returns:
            predictions: Model predictions [batch, 1]
        """
        inputs = {
            'drug_encoded': drug_encoded,
            'se_indices': se_indices,
            'drug_mask': drug_mask,
            'se_mask': se_mask
        }

        outputs = self.session.run(self.output_names, inputs)
        return outputs[0]

    def predict_batch(
            self,
            drug_smiles: List[str],
            se_ids: List[int],
            smiles_encoder,
            se_index: np.ndarray,
            se_mask: np.ndarray
    ) -> np.ndarray:
        """
        Predict for a batch of drug-side effect pairs

        Args:
            drug_smiles: List of drug SMILES strings
            se_ids: List of side effect IDs
            smiles_encoder: SMILES encoder instance
            se_index: Side effect index array [num_se, seq_len]
            se_mask: Side effect mask array [num_se, seq_len]

        Returns:
            predictions: Model predictions [batch]
        """
        batch_size = len(drug_smiles)

        # Encode drugs
        drug_encoded_list = []
        drug_mask_list = []

        for smile in drug_smiles:
            encoded, mask = smiles_encoder.encode(smile)
            drug_encoded_list.append(encoded)
            drug_mask_list.append(mask)

        drug_encoded = np.stack(drug_encoded_list, axis=0)
        drug_mask_np = np.stack(drug_mask_list, axis=0)

        # Get SE encodings
        se_indices_np = np.stack([se_index[se_id] for se_id in se_ids], axis=0)
        se_mask_np = np.stack([se_mask[se_id] for se_id in se_ids], axis=0)

        # Predict
        predictions = self.predict(drug_encoded, se_indices_np, drug_mask_np, se_mask_np)
        return predictions.squeeze()

    def benchmark(
            self,
            batch_size: int = 128,
            seq_len: int = 50,
            num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference speed

        Args:
            batch_size: Batch size for testing
            seq_len: Sequence length
            num_iterations: Number of iterations

        Returns:
            Dict with benchmark results
        """
        logger.info(f"Benchmarking with batch_size={batch_size}, {num_iterations} iterations...")

        # Create dummy inputs
        drug_encoded = np.random.randint(0, 2586, (batch_size, seq_len), dtype=np.int64)
        se_indices = np.random.randint(0, 2586, (batch_size, seq_len), dtype=np.int64)
        drug_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        se_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        # Warmup
        for _ in range(10):
            self.predict(drug_encoded, se_indices, drug_mask, se_mask)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            self.predict(drug_encoded, se_indices, drug_mask, se_mask)
            times.append(time.time() - start)

        times = np.array(times) * 1000  # Convert to ms

        results = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'throughput': float(batch_size / (np.mean(times) / 1000))
        }

        logger.info(f"\nBenchmark Results:")
        logger.info(f"  Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        logger.info(f"  Min: {results['min_ms']:.2f} ms")
        logger.info(f"  Max: {results['max_ms']:.2f} ms")
        logger.info(f"  Throughput: {results['throughput']:.0f} samples/sec\n")

        return results


def compare_pytorch_onnx(
        pytorch_checkpoint: str,
        onnx_path: str,
        config_name: str = 'fast',
        num_samples: int = 100
):
    """Compare PyTorch and ONNX model outputs"""
    from model import create_model
    from config import get_config

    logger.info("Comparing PyTorch and ONNX models...")

    # Load PyTorch model
    config = get_config(config_name)
    pytorch_model = create_model(config.model, device='cpu')
    checkpoint = torch.load(pytorch_checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()

    # Load ONNX model
    onnx_model = ONNXInference(onnx_path)

    # Generate random inputs
    drug_encoded = np.random.randint(0, 2586, (num_samples, 50), dtype=np.int64)
    se_indices = np.random.randint(0, 2586, (num_samples, 50), dtype=np.int64)
    drug_mask = np.ones((num_samples, 50), dtype=np.int64)
    se_mask = np.ones((num_samples, 50), dtype=np.int64)

    # PyTorch inference
    with torch.no_grad():
        pytorch_output, _, _ = pytorch_model(
            torch.from_numpy(drug_encoded),
            torch.from_numpy(se_indices),
            torch.from_numpy(drug_mask),
            torch.from_numpy(se_mask)
        )
        pytorch_output = pytorch_output.numpy()

    # ONNX inference
    onnx_output = onnx_model.predict(drug_encoded, se_indices, drug_mask, se_mask)

    # Compare
    diff = np.abs(pytorch_output - onnx_output)

    logger.info(f"\nComparison Results:")
    logger.info(f"  Max difference: {diff.max():.2e}")
    logger.info(f"  Mean difference: {diff.mean():.2e}")
    logger.info(f"  Std difference: {diff.std():.2e}")

    if diff.max() < 1e-5:
        logger.info("  ✓ Models match perfectly!")
    elif diff.max() < 1e-3:
        logger.info("  ✓ Models match closely (acceptable)")
    else:
        logger.warning("  ⚠ Models differ significantly!")

    return diff.max()


def test_real_predictions(
        onnx_path: str,
        data_dir: str = 'data/processed',
        num_samples: int = 10
):
    """Test ONNX model on real data"""
    logger.info(f"Testing ONNX model on real data...")

    # Load preprocessing assets
    assets = load_preprocessing_assets(data_dir)

    # Create SMILES encoder
    smiles_encoder = CachedSMILESEncoder(
        vocab_path=assets['vocab_file'],
        subword_map_path=assets['subword_map_file']
    )

    # Load ONNX model
    onnx_model = ONNXInference(onnx_path)

    # Load some real data
    import pickle
    with open(Path(data_dir) / 'drug_side.pkl', 'rb') as f:
        data = pickle.load(f)

    df = data['df_data']

    # Sample some pairs
    samples = df.sample(n=num_samples)

    drug_smiles = samples['Drug_smile'].tolist()
    se_ids = samples['SE_id'].astype(int).tolist()
    labels = samples['Label'].tolist()

    # Predict
    predictions = onnx_model.predict_batch(
        drug_smiles,
        se_ids,
        smiles_encoder,
        assets['se_index'],
        assets['se_mask']
    )

    # Print results
    logger.info(f"\nSample Predictions:")
    logger.info(f"{'Drug':<20} {'SE_ID':<10} {'Label':<10} {'Prediction':<10}")
    logger.info("=" * 50)

    for i, (smile, se_id, label, pred) in enumerate(zip(drug_smiles, se_ids, labels, predictions)):
        drug_abbr = smile[:17] + '...' if len(smile) > 20 else smile
        logger.info(f"{drug_abbr:<20} {se_id:<10} {label:<10.0f} {pred:<10.4f}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Test ONNX model')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--checkpoint', type=str,
                        help='PyTorch checkpoint for comparison')
    parser.add_argument('--config', type=str, default='fast',
                        help='Configuration name')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Processed data directory')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with PyTorch model')
    parser.add_argument('--test', action='store_true',
                        help='Test on real data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for benchmark')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of iterations for benchmark')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for ONNX inference')

    args = parser.parse_args()

    # Execution providers
    if args.gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # Benchmark
    if args.benchmark:
        model = ONNXInference(args.onnx, providers=providers)
        model.benchmark(
            batch_size=args.batch_size,
            num_iterations=args.num_iterations
        )

    # Compare with PyTorch
    if args.compare:
        if not args.checkpoint:
            logger.error("--checkpoint required for comparison")
            exit(1)
        compare_pytorch_onnx(
            args.checkpoint,
            args.onnx,
            config_name=args.config
        )

    # Test on real data
    if args.test:
        test_real_predictions(
            args.onnx,
            data_dir=args.data_dir
        )

    # If no action specified, run all tests
    if not (args.benchmark or args.compare or args.test):
        logger.info("Running all tests...")

        model = ONNXInference(args.onnx, providers=providers)
        model.benchmark(batch_size=args.batch_size)

        if args.checkpoint:
            compare_pytorch_onnx(args.checkpoint, args.onnx, args.config)

        test_real_predictions(args.onnx, args.data_dir)


if __name__ == '__main__':
    main()