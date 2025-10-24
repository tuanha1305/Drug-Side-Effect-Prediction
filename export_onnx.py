"""
Export trained model to ONNX format for production deployment
Supports both single model and ensemble export
"""

import torch
import torch.onnx
import argparse
import logging
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict
import json

from model import DrugSideEffectModel, create_model
from config import Config, get_config
from smiles_encoder import CachedSMILESEncoder
from preprocessing import load_side_effect_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export PyTorch model to ONNX format"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Config,
        output_dir: Path,
        opset_version: int = 14
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.opset_version = opset_version
        
    def prepare_dummy_inputs(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
        """
        Create dummy inputs for ONNX export
        
        Args:
            batch_size: Batch size for dummy inputs
            
        Returns:
            Tuple of dummy tensors (drug, se, drug_mask, se_mask)
        """
        device = next(self.model.parameters()).device
        
        # Create dummy inputs with correct shapes
        drug_encoded = torch.randint(
            0, self.config.model.vocab_size,
            (batch_size, self.config.model.max_drug_len),
            dtype=torch.long,
            device=device
        )
        
        se_indices = torch.randint(
            0, self.config.model.vocab_size,
            (batch_size, self.config.model.max_se_len),
            dtype=torch.long,
            device=device
        )
        
        drug_mask = torch.ones(
            (batch_size, self.config.model.max_drug_len),
            dtype=torch.long,
            device=device
        )
        
        se_mask = torch.ones(
            (batch_size, self.config.model.max_se_len),
            dtype=torch.long,
            device=device
        )
        
        return drug_encoded, se_indices, drug_mask, se_mask
    
    def export_model(
        self,
        model_name: str = "drug_side_effect_model",
        dynamic_axes: bool = True,
        simplify: bool = True
    ) -> Path:
        """
        Export model to ONNX format
        
        Args:
            model_name: Name for the exported model
            dynamic_axes: Whether to use dynamic axes for variable batch size
            simplify: Whether to simplify the ONNX model
            
        Returns:
            Path to exported ONNX model
        """
        logger.info("Starting ONNX export...")
        
        # Set model to eval mode
        self.model.eval()
        
        # Prepare dummy inputs
        dummy_inputs = self.prepare_dummy_inputs(batch_size=1)
        
        # Define input and output names
        input_names = ['drug_encoded', 'se_indices', 'drug_mask', 'se_mask']
        output_names = ['prediction', 'drug_attention', 'se_attention']
        
        # Define dynamic axes if requested
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'drug_encoded': {0: 'batch_size'},
                'se_indices': {0: 'batch_size'},
                'drug_mask': {0: 'batch_size'},
                'se_mask': {0: 'batch_size'},
                'prediction': {0: 'batch_size'},
                'drug_attention': {0: 'batch_size'},
                'se_attention': {0: 'batch_size'}
            }
        
        # Output path
        onnx_path = self.output_dir / f"{model_name}.onnx"
        
        # Export to ONNX
        logger.info(f"Exporting to {onnx_path}...")
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_inputs,
                str(onnx_path),
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_dict,
                verbose=False
            )
        
        logger.info(f"✓ Model exported to {onnx_path}")
        
        # Simplify ONNX model if requested
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                
                logger.info("Simplifying ONNX model...")
                model_onnx = onnx.load(str(onnx_path))
                model_simp, check = onnx_simplify(model_onnx)
                
                if check:
                    onnx.save(model_simp, str(onnx_path))
                    logger.info("✓ Model simplified successfully")
                else:
                    logger.warning("⚠ Simplification check failed, using original model")
                    
            except ImportError:
                logger.warning("⚠ onnx-simplifier not installed, skipping simplification")
                logger.info("  Install with: pip install onnx-simplifier")
        
        # Verify the exported model
        self.verify_onnx_model(onnx_path)
        
        # Save model metadata
        self.save_metadata(model_name)
        
        return onnx_path
    
    def verify_onnx_model(self, onnx_path: Path):
        """Verify the exported ONNX model"""
        try:
            import onnx
            
            logger.info("Verifying ONNX model...")
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            logger.info("✓ ONNX model is valid")
            
            # Print model info
            logger.info(f"  Opset version: {model.opset_import[0].version}")
            logger.info(f"  Inputs: {len(model.graph.input)}")
            logger.info(f"  Outputs: {len(model.graph.output)}")
            
        except ImportError:
            logger.warning("⚠ onnx package not installed, skipping verification")
        except Exception as e:
            logger.error(f"✗ ONNX model verification failed: {e}")
    
    def save_metadata(self, model_name: str):
        """Save model metadata for inference"""
        metadata = {
            'model_name': model_name,
            'vocab_size': self.config.model.vocab_size,
            'num_side_effects': self.config.model.num_side_effects,
            'max_drug_len': self.config.model.max_drug_len,
            'max_se_len': self.config.model.max_se_len,
            'embedding_dim': self.config.model.embedding_dim,
            'num_encoder_layers': self.config.model.num_encoder_layers,
            'num_attention_heads': self.config.model.num_attention_heads,
            'input_names': ['drug_encoded', 'se_indices', 'drug_mask', 'se_mask'],
            'output_names': ['prediction', 'drug_attention', 'se_attention'],
            'opset_version': self.opset_version
        }
        
        metadata_path = self.output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved to {metadata_path}")
    
    def test_onnx_inference(self, onnx_path: Path) -> bool:
        """
        Test ONNX model inference
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            True if test passed
        """
        try:
            import onnxruntime as ort
            
            logger.info("Testing ONNX inference...")
            
            # Create session
            session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            
            # Prepare test inputs
            dummy_inputs = self.prepare_dummy_inputs(batch_size=2)
            input_dict = {
                'drug_encoded': dummy_inputs[0].cpu().numpy(),
                'se_indices': dummy_inputs[1].cpu().numpy(),
                'drug_mask': dummy_inputs[2].cpu().numpy(),
                'se_mask': dummy_inputs[3].cpu().numpy()
            }
            
            # Run inference
            outputs = session.run(None, input_dict)
            
            logger.info("✓ ONNX inference test passed")
            logger.info(f"  Output shapes: {[out.shape for out in outputs]}")
            
            # Compare with PyTorch
            self.model.eval()
            with torch.no_grad():
                torch_outputs = self.model(*dummy_inputs)
                torch_pred = torch_outputs[0].cpu().numpy()
            
            onnx_pred = outputs[0]
            diff = np.abs(torch_pred - onnx_pred).max()
            logger.info(f"  Max difference from PyTorch: {diff:.6f}")
            
            if diff < 1e-4:
                logger.info("✓ ONNX output matches PyTorch (diff < 1e-4)")
                return True
            else:
                logger.warning(f"⚠ ONNX output differs from PyTorch (diff = {diff})")
                return False
                
        except ImportError:
            logger.warning("⚠ onnxruntime not installed, skipping inference test")
            logger.info("  Install with: pip install onnxruntime")
            return False
        except Exception as e:
            logger.error(f"✗ ONNX inference test failed: {e}")
            return False


def export_single_model(
    checkpoint_path: str,
    output_dir: str,
    config_name: str = 'default',
    model_name: Optional[str] = None,
    dynamic_axes: bool = True,
    simplify: bool = True,
    test_inference: bool = True
) -> Path:
    """
    Export a single trained model to ONNX
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save ONNX model
        config_name: Config name to use
        model_name: Custom model name (default: uses checkpoint name)
        dynamic_axes: Use dynamic axes for variable batch size
        simplify: Simplify ONNX model
        test_inference: Test ONNX inference
        
    Returns:
        Path to exported ONNX model
    """
    logger.info("=" * 60)
    logger.info("Exporting Single Model to ONNX")
    logger.info("=" * 60)
    
    # Load config
    config = get_config(config_name)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config.model, device=device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("✓ Model loaded successfully")
    
    # Determine model name
    if model_name is None:
        model_name = Path(checkpoint_path).stem
    
    # Create exporter
    exporter = ONNXExporter(model, config, output_dir)
    
    # Export model
    onnx_path = exporter.export_model(
        model_name=model_name,
        dynamic_axes=dynamic_axes,
        simplify=simplify
    )
    
    # Test inference
    if test_inference:
        exporter.test_onnx_inference(onnx_path)
    
    logger.info("=" * 60)
    logger.info("Export completed successfully!")
    logger.info("=" * 60)
    
    return onnx_path


def export_ensemble(
    checkpoint_dir: str,
    output_dir: str,
    config_name: str = 'default',
    fold_pattern: str = 'fold_*_best.pth',
    ensemble_name: str = 'ensemble',
    export_individual: bool = True
):
    """
    Export ensemble of models (all folds) to ONNX
    
    Args:
        checkpoint_dir: Directory containing fold checkpoints
        output_dir: Directory to save ONNX models
        config_name: Config name
        fold_pattern: Pattern to match fold checkpoints
        ensemble_name: Name prefix for ensemble models
        export_individual: Export individual fold models
    """
    logger.info("=" * 60)
    logger.info("Exporting Ensemble to ONNX")
    logger.info("=" * 60)
    
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all fold checkpoints
    checkpoints = sorted(checkpoint_dir.glob(fold_pattern))
    
    if not checkpoints:
        logger.error(f"No checkpoints found matching pattern: {fold_pattern}")
        return
    
    logger.info(f"Found {len(checkpoints)} checkpoints")
    
    # Export each fold
    exported_models = []
    for i, ckpt_path in enumerate(checkpoints):
        logger.info(f"\n--- Exporting Fold {i} ---")
        
        model_name = f"{ensemble_name}_fold_{i}"
        
        try:
            onnx_path = export_single_model(
                checkpoint_path=str(ckpt_path),
                output_dir=str(output_dir),
                config_name=config_name,
                model_name=model_name,
                test_inference=(i == 0)  # Only test first fold
            )
            exported_models.append(onnx_path)
            
        except Exception as e:
            logger.error(f"Failed to export fold {i}: {e}")
    
    # Save ensemble metadata
    ensemble_metadata = {
        'ensemble_name': ensemble_name,
        'num_models': len(exported_models),
        'models': [str(p.name) for p in exported_models],
        'inference_method': 'average',  # or 'voting'
        'description': 'Ensemble of models trained on different folds'
    }
    
    metadata_path = output_dir / f"{ensemble_name}_ensemble_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)
    
    logger.info(f"\n✓ Ensemble metadata saved to {metadata_path}")
    logger.info("=" * 60)
    logger.info(f"Exported {len(exported_models)}/{len(checkpoints)} models")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX format')
    
    # Mode
    parser.add_argument('--mode', type=str, choices=['single', 'ensemble'],
                       default='single', help='Export mode')
    
    # Single model export
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint (for single mode)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Custom model name')
    
    # Ensemble export
    parser.add_argument('--checkpoint_dir', type=str,
                       help='Directory with fold checkpoints (for ensemble mode)')
    parser.add_argument('--fold_pattern', type=str, default='fold_*_best.pth',
                       help='Pattern to match fold checkpoints')
    parser.add_argument('--ensemble_name', type=str, default='ensemble',
                       help='Ensemble name prefix')
    
    # Common options
    parser.add_argument('--output_dir', type=str, default='onnx_models',
                       help='Output directory for ONNX models')
    parser.add_argument('--config', type=str, default='default',
                       help='Config name')
    parser.add_argument('--no_dynamic_axes', action='store_true',
                       help='Disable dynamic axes')
    parser.add_argument('--no_simplify', action='store_true',
                       help='Skip ONNX simplification')
    parser.add_argument('--no_test', action='store_true',
                       help='Skip inference test')
    parser.add_argument('--opset_version', type=int, default=14,
                       help='ONNX opset version')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.checkpoint:
            parser.error("--checkpoint is required for single mode")
        
        export_single_model(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            config_name=args.config,
            model_name=args.model_name,
            dynamic_axes=not args.no_dynamic_axes,
            simplify=not args.no_simplify,
            test_inference=not args.no_test
        )
        
    elif args.mode == 'ensemble':
        if not args.checkpoint_dir:
            parser.error("--checkpoint_dir is required for ensemble mode")
        
        export_ensemble(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            config_name=args.config,
            fold_pattern=args.fold_pattern,
            ensemble_name=args.ensemble_name
        )


if __name__ == '__main__':
    main()