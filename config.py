"""
Configuration file for drug side effect prediction model
Optimized for PyTorch 2.x
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path
import torch


@dataclass
class DataConfig:
    """Data configuration"""
    # Paths
    data_root: Path = Path("data")
    raw_data_dir: Path = data_root / "raw"
    processed_data_dir: Path = data_root / "processed"
    cache_dir: Path = data_root / "cache"
    
    # Files
    drug_smiles_file: str = "drug_SMILES_750.csv"
    vocab_file: str = "drug_codes_chembl_freq_1500.txt"
    subword_map_file: str = "subword_units_map_chembl_freq_1500.csv"
    drug_side_pkl: str = "drug_side.pkl"
    
    # Dataset parameters
    max_drug_len: int = 50
    max_se_len: int = 50
    vocab_size: int = 2586
    num_side_effects: int = 994
    num_drugs: int = 750
    
    # Preprocessing
    percentile_threshold: float = 95.0
    top_k_substructures: int = 50
    
    # Cross-validation
    n_folds: int = 10
    random_state: int = 1
    
    # Negative sampling
    addition_negative_strategy: str = 'all'  # 'all' or number
    

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Vocabulary and sizes
    vocab_size: int = 2586
    num_side_effects: int = 994
    max_drug_len: int = 50
    max_se_len: int = 50
    
    # Embedding
    embedding_dim: int = 200
    max_position_embeddings: int = 500
    dropout_rate: float = 0.1
    
    # Transformer Encoder
    num_encoder_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 512
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # Decoder MLP
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 64, 32])
    decoder_input_dim: int = 6912  # Calculated from interaction layer
    decoder_output_dim: int = 1
    decoder_dropout: float = 0.3
    use_batch_norm: bool = True
    
    # Interaction layer
    use_cross_attention: bool = False
    conv_in_channels: int = 1
    conv_out_channels: int = 3
    conv_kernel_size: int = 3
    conv_padding: int = 0
    
    # Flash attention (PyTorch 2.x optimization)
    use_flash_attention: bool = True
    use_sdpa: bool = True  # Scaled Dot Product Attention
    
    # Gradient checkpointing for memory efficiency
    use_gradient_checkpointing: bool = False
    

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # Optimizer type
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    use_fused_optimizer: bool = True  # PyTorch 2.x fused optimizer
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "linear", "step"
    warmup_steps: int = 500
    warmup_ratio: float = 0.1
    
    # Training parameters
    num_epochs: int = 200
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision training (PyTorch 2.x AMP)
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"
    
    # PyTorch 2.x compile
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    
    # Checkpointing
    save_checkpoint_every: int = 50
    save_best_only: bool = True
    monitor_metric: str = "auc"  # "auc", "aupr", "loss"
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1  # Evaluate every N epochs
    
    # Seed for reproducibility
    seed: int = 42


@dataclass
class DataLoaderConfig:
    """DataLoader configuration optimized for PyTorch 2.x"""
    # Worker settings
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True  # PyTorch 2.x feature
    prefetch_factor: int = 2
    
    # Shuffle
    shuffle_train: bool = True
    shuffle_val: bool = False
    shuffle_test: bool = False
    
    # Drop last
    drop_last_train: bool = False
    drop_last_val: bool = False
    

@dataclass
class PathConfig:
    """Path configuration"""
    # Output directories
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    result_dir: Path = Path("results")
    tensorboard_dir: Path = Path("logs/tensorboard")
    
    # Create directories if not exist
    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration combining all sub-configs"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_name: str = "cuda:0"
    
    # Experiment name
    experiment_name: str = "drug_side_effect_prediction"
    
    def __post_init__(self):
        """Validate and adjust configurations"""
        # Auto-detect optimal num_workers
        if self.dataloader.num_workers == 0:
            import os
            self.dataloader.num_workers = min(4, os.cpu_count() or 1)
        
        # Disable AMP and compile if on CPU
        if self.device == "cpu":
            if self.training.use_amp:
                print("Info: Disabling AMP (not supported on CPU)")
                self.training.use_amp = False
            if self.training.compile_model:
                print("Info: Disabling torch.compile on CPU (may cause warnings)")
                self.training.compile_model = False
        
        # Check CUDA capability for bfloat16
        if self.training.amp_dtype == "bfloat16" and torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] < 8:
                print("Warning: bfloat16 requires Ampere GPU or newer, falling back to float16")
                self.training.amp_dtype = "float16"
        
        # Disable persistent_workers if num_workers is 0
        if self.dataloader.num_workers == 0:
            self.dataloader.persistent_workers = False
        
        # Suppress torch.compile warnings on CPU
        if self.device == "cpu":
            import warnings
            warnings.filterwarnings('ignore', message='.*cudagraph.*')
    
    def to_dict(self):
        """Convert config to dictionary"""
        def convert_value(v):
            """Convert Path and other non-serializable objects to strings"""
            if isinstance(v, Path):
                return str(v)
            elif isinstance(v, (list, tuple)):
                return [convert_value(item) for item in v]
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            return v
        
        return {
            'data': {k: convert_value(v) for k, v in self.data.__dict__.items()},
            'model': {k: convert_value(v) for k, v in self.model.__dict__.items()},
            'training': {k: convert_value(v) for k, v in self.training.__dict__.items()},
            'dataloader': {k: convert_value(v) for k, v in self.dataloader.__dict__.items()},
            'paths': {k: str(v) for k, v in self.paths.__dict__.items()},
            'device': self.device,
            'cuda_name': self.cuda_name,
            'experiment_name': self.experiment_name
        }
    
    def save(self, path: str):
        """Save config to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, path: str):
        """Load config from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        if "data" in config_dict and isinstance(config_dict["data"], dict):
            config_dict["data"] = DataConfig(**config_dict["data"])

        if "model" in config_dict and isinstance(config_dict["model"], dict):
            config_dict["model"] = ModelConfig(**config_dict["model"])

        if "training" in config_dict and isinstance(config_dict["training"], dict):
            config_dict["training"] = TrainingConfig(**config_dict["training"])

        if "dataloader" in config_dict and isinstance(config_dict["dataloader"], dict):
            config_dict["dataloader"] = DataLoaderConfig(**config_dict["dataloader"])

        if "paths" in config_dict and isinstance(config_dict["paths"], dict):
            paths_dict = config_dict["paths"]
            for key, value in paths_dict.items():
                paths_dict[key] = Path(value)  # convert str → Path
            config_dict["paths"] = PathConfig(**paths_dict)

            # === Initialize main Config ===
        return cls(**config_dict)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def get_debug_config() -> Config:
    """Get debug configuration for quick testing"""
    config = Config()
    config.training.num_epochs = 5
    config.training.batch_size = 32
    config.data.n_folds = 2
    config.training.log_interval = 10
    config.dataloader.num_workers = 0
    config.training.compile_model = False
    return config


def get_fast_config() -> Config:
    """Get configuration optimized for speed"""
    config = Config()
    config.training.compile_model = True
    config.training.compile_mode = "max-autotune"
    config.training.use_amp = True
    config.model.use_flash_attention = True
    config.model.use_sdpa = True
    config.dataloader.num_workers = 8
    config.dataloader.prefetch_factor = 4
    return config


def get_memory_efficient_config() -> Config:
    """Get configuration optimized for low memory"""
    config = Config()
    config.training.batch_size = 64
    config.training.gradient_accumulation_steps = 2
    config.model.use_gradient_checkpointing = True
    config.training.use_amp = True
    config.dataloader.num_workers = 2
    return config


if __name__ == "__main__":
    # Test configurations
    config = get_default_config()
    print("Default Config:")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Num workers: {config.dataloader.num_workers}")
    print(f"Use AMP: {config.training.use_amp}")
    print(f"Compile model: {config.training.compile_model}")
    
    # Save config
    config.save("config_default.json")
    print("\nConfig saved to config_default.json")