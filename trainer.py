"""
Training module for drug side effect prediction
Optimized for PyTorch 2.x with mixed precision, gradient accumulation, and torch.compile
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import time
from tqdm import tqdm
import logging

from config import Config
from model import DrugSideEffectModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for drug side effect prediction model
    Optimized for PyTorch 2.x
    """

    def __init__(
            self,
            model: DrugSideEffectModel,
            config: Config,
            train_loader: DataLoader,
            val_loader: DataLoader,
            fold: int = 0
    ):
        """
        Initialize trainer

        Args:
            model: Model to train
            config: Configuration object
            train_loader: Training data loader
            val_loader: Validation data loader
            fold: Fold number for cross-validation
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold = fold

        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Optimization components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler() if config.training.use_scheduler else None
        self.criterion = self._create_criterion()

        # Mixed precision training
        self.use_amp = config.training.use_amp
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps

        # Compile model (PyTorch 2.x)
        if config.training.compile_model and hasattr(torch, 'compile'):
            logger.info(f"Compiling model with mode: {config.training.compile_mode}")
            self.model = torch.compile(
                self.model,
                mode=config.training.compile_mode
            )

        # Tracking
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.train_losses = []
        self.val_metrics = []

        # Early stopping
        self.patience_counter = 0

        # TensorBoard
        log_dir = config.paths.tensorboard_dir / f"fold_{fold}"
        self.writer = SummaryWriter(log_dir)

        # Checkpoint directory
        self.checkpoint_dir = config.paths.checkpoint_dir / f"fold_{fold}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized for fold {fold}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        config = self.config.training

        if config.optimizer.lower() == 'adam':
            if config.use_fused_optimizer and self.device.type == 'cuda':
                optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=config.learning_rate,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay,
                    fused=True
                )
            else:
                optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=config.learning_rate,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay
                )
        elif config.optimizer.lower() == 'adamw':
            if config.use_fused_optimizer and self.device.type == 'cuda':
                optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=config.learning_rate,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay,
                    fused=True
                )
            else:
                optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=config.learning_rate,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay
                )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        logger.info(f"Created optimizer: {config.optimizer}")
        return optimizer

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        config = self.config.training

        if config.scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.learning_rate * 0.01
            )
        elif config.scheduler_type == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=config.num_epochs
            )
        elif config.scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.num_epochs // 3,
                gamma=0.1
            )
        else:
            return None

        logger.info(f"Created scheduler: {config.scheduler_type}")
        return scheduler

    def _create_criterion(self) -> nn.Module:
        """Create loss function"""
        # MSE Loss for regression
        return nn.BCEWithLogitsLoss()

    def train_epoch(self) -> float:
        """
        Train for one epoch

        Returns:
            avg_loss: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}",
            leave=True
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            drug, se, drug_mask, se_mask, label = batch

            # Move to device
            drug = drug.to(self.device)
            se = se.to(self.device)
            drug_mask = drug_mask.to(self.device)
            se_mask = se_mask.to(self.device)
            label = label.to(self.device).float()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    output, _, _ = self.model(drug, se, drug_mask, se_mask)
                    loss = self.criterion(output.squeeze(), label)
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.training.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm
                        )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training
                output, _, _ = self.model(drug, se, drug_mask, se_mask)
                loss = self.criterion(output.squeeze(), label)
                loss = loss / self.gradient_accumulation_steps

                # ===== Debug block =====
                if batch_idx % 100 == 0 or loss.item() < 0:
                    logger.warning(
                        f"[DEBUG] Epoch {self.current_epoch + 1}, Batch {batch_idx}: "
                        f"Device={self.device}, Dtype={output.dtype}, "
                        f"Output Range=({output.min().item():.3f}, {output.max().item():.3f}), "
                        f"Mean={output.mean().item():.3f}, Loss={loss.item():.6f}"
                    )
                # ========================

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.training.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Accumulate loss (unscaled)
            total_loss += loss.item() * self.gradient_accumulation_steps

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # Log to tensorboard
            if batch_idx % self.config.training.log_interval == 0:
                global_step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar(
                    'train/batch_loss',
                    loss.item() * self.gradient_accumulation_steps,
                    global_step
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set

        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            drug, se, drug_mask, se_mask, label = batch

            # Move to device
            drug = drug.to(self.device)
            se = se.to(self.device)
            drug_mask = drug_mask.to(self.device)
            se_mask = se_mask.to(self.device)
            label = label.to(self.device).float()

            # Forward pass
            if self.use_amp:
                with autocast():
                    output, _, _ = self.model(drug, se, drug_mask, se_mask)
                    loss = self.criterion(output.squeeze(), label)
            else:
                output, _, _ = self.model(drug, se, drug_mask, se_mask)
                loss = self.criterion(output.squeeze(), label)

            total_loss += loss.item()

            # Convert logits -> probabilities for metrics
            probs = torch.sigmoid(output)

            # Collect predictions and labels
            all_probs.append(probs.squeeze().cpu().numpy())
            all_labels.append(label.cpu().numpy())

        # Concatenate all batches
        avg_loss = total_loss / len(self.val_loader)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # ===== Metrics on probability scale =====
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            roc_auc_score,
            average_precision_score
        )
        from scipy.stats import pearsonr, spearmanr

        rmse = np.sqrt(mean_squared_error(all_labels, all_probs))
        mae = mean_absolute_error(all_labels, all_probs)

        # Only calculate correlation if there's variance
        if len(np.unique(all_labels)) > 1 and len(np.unique(all_probs)) > 1:
            pearson, _ = pearsonr(all_labels, all_probs)
            spearman, _ = spearmanr(all_labels, all_probs)
            try:
                auc = roc_auc_score(all_labels, all_probs)
                aupr = average_precision_score(all_labels, all_probs)
            except ValueError:
                auc, aupr = 0.0, 0.0
        else:
            pearson, spearman, auc, aupr = 0.0, 0.0, 0.0, 0.0

        metrics = {
            'loss': avg_loss,
            'rmse': rmse,
            'mae': mae,
            'pearson': pearson,
            'spearman': spearman,
            'auc': auc,
            'aupr': aupr
        }

        return metrics

    def train(self):
        """
        Main training loop
        """
        logger.info(f"Starting training for {self.config.training.num_epochs} epochs...")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        start_time = time.time()

        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            if (epoch + 1) % self.config.training.eval_interval == 0:
                val_metrics = self.validate()
                self.val_metrics.append(val_metrics)

                # Log metrics
                logger.info("=" * 120)
                logger.info(
                    f"[Epoch {epoch + 1:03d}/{self.config.training.num_epochs}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"RMSE: {val_metrics['rmse']:.4f} | "
                    f"MAE: {val_metrics['mae']:.4f} | "
                    f"Pearson: {val_metrics['pearson']:.4f} | "
                    f"Spearman: {val_metrics['spearman']:.4f} | "
                    f"AUC: {val_metrics.get('auc', 0.0):.4f} | "
                    f"AUPR: {val_metrics.get('aupr', 0.0):.4f}"
                )
                logger.info("=" * 120)

                # TensorBoard logging
                self.writer.add_scalar('train/loss', train_loss, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                self.writer.add_scalar(
                    'train/lr',
                    self.optimizer.param_groups[0]['lr'],
                    epoch
                )

                # Save checkpoint
                monitor_metric = val_metrics.get(
                    self.config.training.monitor_metric,
                    val_metrics['loss']
                )

                is_best = False
                if self.config.training.monitor_metric == 'loss':
                    is_best = monitor_metric < -self.best_metric
                else:
                    is_best = monitor_metric > self.best_metric

                if is_best:
                    self.best_metric = monitor_metric if self.config.training.monitor_metric != 'loss' else -monitor_metric
                    self.patience_counter = 0

                    if self.config.training.save_best_only:
                        self.save_checkpoint(is_best=True)
                        logger.info(
                            f"✓ Saved best model with {self.config.training.monitor_metric}: {monitor_metric:.4f}")
                else:
                    self.patience_counter += 1

                # Regular checkpoint
                if not self.config.training.save_best_only and (
                        epoch + 1) % self.config.training.save_checkpoint_every == 0:
                    self.save_checkpoint(is_best=False, epoch=epoch)

                # Early stopping
                if self.config.training.early_stopping:
                    if self.patience_counter >= self.config.training.patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

        # Training completed
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time / 60:.2f} minutes")
        logger.info(f"Best {self.config.training.monitor_metric}: {self.best_metric:.4f}")

        # Close tensorboard writer
        self.writer.close()

    def save_checkpoint(self, is_best: bool = False, epoch: Optional[int] = None):
        """
        Save model checkpoint

        Args:
            is_best: Whether this is the best model
            epoch: Epoch number (for regular checkpoints)
        """

        model_to_save = (
            self.model._orig_mod
            if hasattr(self.model, "_orig_mod") else
            self.model.module
            if hasattr(self.model, "module") else
            self.model
        )

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


if __name__ == "__main__":
    # Test trainer
    from config import get_default_config, ModelConfig
    from model import create_model
    from torch.utils.data import TensorDataset

    print("=" * 60)
    print("Testing Trainer")
    print("=" * 60)

    # Get config
    config = get_default_config()
    config.training.num_epochs = 3
    config.training.log_interval = 10
    config.dataloader.num_workers = 0

    # Create dummy model
    model_config = ModelConfig()
    model_config.vocab_size = 2586
    model = create_model(model_config, device=config.device)

    # Create dummy data
    n_train = 100
    n_val = 20
    seq_len = 50

    train_dataset = TensorDataset(
        torch.randint(0, 2586, (n_train, seq_len)),
        torch.randint(0, 2586, (n_train, seq_len)),
        torch.ones((n_train, seq_len)),
        torch.ones((n_train, seq_len)),
        torch.rand(n_train)
    )

    val_dataset = TensorDataset(
        torch.randint(0, 2586, (n_val, seq_len)),
        torch.randint(0, 2586, (n_val, seq_len)),
        torch.ones((n_val, seq_len)),
        torch.ones((n_val, seq_len)),
        torch.rand(n_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        fold=0
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    print("\n✓ Trainer test completed!")