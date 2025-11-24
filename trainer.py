"""
Training module for drug side effect prediction
Aligned with HSTrans paper:
- Uses MSE Loss (Eq. 16)
- Calculates all metrics including Overlap@N% via Evaluator
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
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from model import DrugSideEffectModel
from losses import MSELoss
from evaluator import Evaluator  # Use the standardized Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for drug side effect prediction model
    """

    def __init__(
            self,
            model: DrugSideEffectModel,
            config: Config,
            train_loader: DataLoader,
            val_loader: DataLoader,
            fold: int = 0
    ):
        """Initialize trainer"""
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

        # FIXED: Use MSELoss as specified in Paper Eq (16)
        # Note: RMSE is used for evaluation metric, but MSE is the training loss
        self.criterion = MSELoss(reduction='mean')

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

        # Initialize Evaluator for consistent metric calculation
        self.evaluator = Evaluator(self.model, device=config.device)

        logger.info(f"Trainer initialized for fold {fold}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer (Adam as per paper)"""
        config = self.config.training

        # Paper uses Adam with lr=1e-4
        if config.optimizer.lower() == 'adam':
            # Use fused Adam if available (faster on CUDA)
            use_fused = config.use_fused_optimizer and self.device.type == 'cuda'
            try:
                optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=config.learning_rate,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay,
                    fused=use_fused
                )
            except:
                # Fallback if fused not supported
                optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=config.learning_rate,
                    betas=config.betas,
                    eps=config.eps,
                    weight_decay=config.weight_decay
                )
        else:
            # Fallback/Alternative
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay
            )

        logger.info(f"Created optimizer: {type(optimizer).__name__}")
        return optimizer

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        config = self.config.training

        # Note: Paper doesn't explicitly mention scheduler, but cosine is standard
        if config.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.learning_rate * 0.01
            )
        return None

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

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

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.config.training.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output, _, _ = self.model(drug, se, drug_mask, se_mask)
                loss = self.criterion(output.squeeze(), label)
                loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.config.training.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.training.max_grad_norm
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            current_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += current_loss

            pbar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            if batch_idx % self.config.training.log_interval == 0:
                global_step = self.current_epoch * num_batches + batch_idx
                self.writer.add_scalar('train/batch_loss', current_loss, global_step)

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model using Evaluator
        Calculates all metrics: RMSE, MAE, SCC, Overlap@N%
        """
        # FIXED: Delegate to Evaluator class to ensure consistency
        # Evaluator handles model.eval(), prediction loop, and metric calc (including Overlap@N)
        metrics = self.evaluator.evaluate(self.val_loader)

        if 'loss' not in metrics:
            if 'mse' in metrics:
                metrics['loss'] = metrics['mse']
            elif 'rmse' in metrics:
                metrics['loss'] = metrics['rmse'] ** 2

        return metrics

    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.config.training.num_epochs} epochs...")

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

                # Log metrics (Expanded to show Overlap metrics)
                log_msg = (
                    f"Epoch {epoch + 1}/{self.config.training.num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
                    f"RMSE: {val_metrics.get('rmse', 0):.4f} | "
                    f"MAE: {val_metrics.get('mae', 0):.4f} | "
                    f"SCC: {val_metrics.get('scc', 0):.4f}"
                )

                # Add Overlap metrics to log if available
                if 'overlap@1%' in val_metrics:
                     log_msg += f" | Ov@1%: {val_metrics['overlap@1%']:.3f}"

                logger.info(log_msg)

                # TensorBoard logging
                self.writer.add_scalar('train/loss', train_loss, epoch)
                for key, value in val_metrics.items():
                    # Group metrics for cleaner tensorboard
                    if 'overlap' in key:
                        self.writer.add_scalar(f'val_overlap/{key}', value, epoch)
                    else:
                        self.writer.add_scalar(f'val/{key}', value, epoch)

                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)

                # Save checkpoint logic
                self._handle_checkpoint(val_metrics, epoch)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time / 60:.2f} minutes")
        logger.info(f"Best {self.config.training.monitor_metric}: {self.best_metric:.4f}")
        self.writer.close()

    def _handle_checkpoint(self, val_metrics: Dict[str, float], epoch: int):
        """Handle checkpoint saving based on metric monitoring"""
        monitor_metric_name = self.config.training.monitor_metric
        # Default fallback if metric not found (e.g. 'auc' might not be in regression metrics)
        if monitor_metric_name not in val_metrics:
            # Fallback to 'scc' or 'rmse' for regression
            monitor_metric_name = 'scc' if 'scc' in val_metrics else 'loss'

        current_val = val_metrics.get(monitor_metric_name, 0.0)

        is_best = False
        # For loss/rmse/mae, lower is better. For others (scc, overlap), higher is better.
        lower_is_better = monitor_metric_name in ['loss', 'rmse', 'mae', 'mse']

        if lower_is_better:
            # We store negative value for "best_metric" to keep logic consistent (maximizing)
            # OR we just implement explicit logic:
            if self.best_metric == float('-inf'): # First run
                self.best_metric = float('inf')

            if current_val < self.best_metric:
                self.best_metric = current_val
                is_best = True
        else:
            if current_val > self.best_metric:
                self.best_metric = current_val
                is_best = True

        if is_best:
            self.patience_counter = 0
            if self.config.training.save_best_only:
                self.save_checkpoint(is_best=True)
                logger.info(f"✓ Saved best model ({monitor_metric_name}: {current_val:.4f})")
        else:
            self.patience_counter += 1

        if not self.config.training.save_best_only and (epoch + 1) % self.config.training.save_checkpoint_every == 0:
            self.save_checkpoint(is_best=False, epoch=epoch)

        if self.config.training.early_stopping and self.patience_counter >= self.config.training.patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            # Hack to stop training loop: set current epoch to max
            self.current_epoch = self.config.training.num_epochs

    def save_checkpoint(self, is_best: bool = False, epoch: Optional[int] = None):
        """Save checkpoint"""
        model_to_save = self.model
        # Unwrap compiled/parallel model
        if hasattr(self.model, "_orig_mod"): model_to_save = self.model._orig_mod
        elif hasattr(self.model, "module"): model_to_save = self.model.module

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
        }

        if self.scheduler: checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler: checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle unwrapped loading
        model_to_load = self.model
        if hasattr(self.model, "_orig_mod"): model_to_load = self.model._orig_mod

        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

if __name__ == "__main__":
    pass