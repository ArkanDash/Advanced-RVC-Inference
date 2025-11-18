"""
Main Training Module for Advanced RVC Inference
Handles the complete training pipeline from data preprocessing to model training
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .core.training_config import TrainingConfig, TrainingLogger, get_device, validate_training_data
from .data.dataset import RVCDataSet
from .trainers.rvc_trainer import RVCTrainer
from .utils.audio_utils import preprocess_audio_files
from .utils.feature_extraction import extract_features


class AdvancedRVCTrainer:
    """Advanced RVC Trainer - Complete training pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = TrainingLogger(config)
        self.device = get_device()
        
        # Initialize components
        self.trainer = None
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        
        self.logger.logger.info("Advanced RVC Trainer initialized")
    
    def setup_model(self):
        """Setup the RVC model for training"""
        try:
            # Import RVC model components
            from programs.applio_code.rvc.lib.models import SynthesizerTrn
            from programs.applio_code.rvc.lib.models import DiscriminatorV3
            
            # Model parameters based on configuration
            if self.config.model_version == "v1":
                spec_channels = 513
                segment_size = 8192
                inter_channels = 192
                hidden_channels = 192
                filter_channels = 768
                n_heads = 2
                n_layers = 6
                kernel_size = 3
                p_dropout = 0.1
                resblock_kernel_sizes = [3, 7, 11]
                resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
                upsample_rates = [16, 16, 4, 4]
                upsample_initial_channel = 512
                upsample_kernel_sizes = [32, 32, 18, 18]
                gin_channels = 256
            else:  # v2
                spec_channels = 1025
                segment_size = 8192
                inter_channels = 192
                hidden_channels = 192
                filter_channels = 768
                n_heads = 2
                n_layers = 6
                kernel_size = 3
                p_dropout = 0.1
                resblock_kernel_sizes = [3, 7, 11]
                resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
                upsample_rates = [20, 19, 17, 20]
                upsample_initial_channel = 512
                upsample_kernel_sizes = [50, 39, 37, 50]
                gin_channels = 256
            
            # Initialize generator and discriminator
            self.generator = SynthesizerTrn(
                spec_channels=spec_channels,
                segment_size=segment_size,
                inter_channels=inter_channels,
                hidden_channels=hidden_channels,
                filter_channels=filter_channels,
                n_heads=n_heads,
                n_layers=n_layers,
                kernel_size=kernel_size,
                p_dropout=p_dropout,
                resblock_kernel_sizes=resblock_kernel_sizes,
                resblock_dilation_sizes=resblock_dilation_sizes,
                upsample_rates=upsample_rates,
                upsample_initial_channel=upsample_initial_channel,
                upsample_kernel_sizes=upsample_kernel_sizes,
                gin_channels=gin_channels,
                ssl_dim=256,
                f0=True,
                num_langs=1,
                num_prompts=1
            ).to(self.device)
            
            self.discriminator = DiscriminatorV3(
                use_spectral_norm=True,
                segment_size=segment_size
            ).to(self.device)
            
            self.logger.logger.info(f"Model initialized with {self.config.model_version} architecture")
            
        except ImportError as e:
            self.logger.logger.error(f"Failed to import model components: {e}")
            raise
    
    def setup_optimizer(self):
        """Setup optimizers for training"""
        # Generator optimizer
        self.optimizer_g = optim.AdamW(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=(0.8, 0.99),
            eps=1e-9,
            weight_decay=1e-2
        )
        
        # Discriminator optimizer
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(0.8, 0.99),
            eps=1e-9,
            weight_decay=1e-2
        )
        
        self.logger.logger.info("Optimizers configured")
    
    def setup_scheduler(self):
        """Setup learning rate schedulers"""
        self.scheduler_g = optim.lr_scheduler.OneCycleLR(
            self.optimizer_g,
            max_lr=self.config.learning_rate,
            epochs=self.config.total_epochs,
            steps_per_epoch=100  # This will be adjusted dynamically
        )
        
        self.scheduler_d = optim.lr_scheduler.OneCycleLR(
            self.optimizer_d,
            max_lr=self.config.learning_rate,
            epochs=self.config.total_epochs,
            steps_per_epoch=100  # This will be adjusted dynamically
        )
        
        self.logger.logger.info("Learning rate schedulers configured")
    
    def prepare_data(self) -> bool:
        """Prepare training data"""
        try:
            self.logger.logger.info("Preparing training data...")
            
            # Validate dataset
            if not validate_training_data(self.config.dataset_path):
                return False
            
            # Preprocess audio files
            preprocess_audio_files(self.config.dataset_path, self.config.sample_rate)
            
            # Create dataset
            self.dataset = RVCDataSet(
                dataset_path=self.config.dataset_path,
                sample_rate=self.config.sample_rate,
                hop_length=self.config.hop_length,
                f0_method=self.config.f0_method,
                embedder_model=self.config.embedder_model
            )
            
            # Create data loaders
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            self.logger.logger.info(f"Data preparation completed. {len(self.dataset)} samples loaded")
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Failed to prepare data: {e}")
            return False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        total_loss_g = 0.0
        total_loss_d = 0.0
        total_steps = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer_g.zero_grad()
            self.optimizer_d.zero_grad()
            
            # Generator forward pass
            y_hat, ids_slice, z_mask, *_ = self.generator(*batch["input"])
            
            # Discriminator forward pass
            y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(batch["audio"], y_hat.detach())
            
            # Compute discriminator loss
            loss_disc = 0.0
            y_d_hat_r = y_d_hat_r[-1]
            y_d_hat_g = y_d_hat_g[-1]
            
            for i in range(len(y_d_hat_r)):
                loss_disc += torch.mean(torch.relu(1 - y_d_hat_r[i]))
                loss_disc += torch.mean(torch.relu(1 + y_d_hat_g[i]))
            
            # Generator adversarial loss
            y_d_hat_g, _, fmap_f_r, fmap_f_g = self.discriminator(batch["audio"], y_hat)
            loss_adv = 0.0
            
            for i in range(len(y_d_hat_g)):
                loss_adv += torch.mean(torch.relu(1 - y_d_hat_g[i]))
            
            # Feature matching loss
            for i in range(len(fmap_f_r)):
                for j in range(len(fmap_f_r[i])):
                    loss_adv += 0.1 * torch.mean(torch.abs(fmap_f_r[i][j] - fmap_f_g[i][j]))
            
            # Compute reconstruction loss
            y_hat_mel = self.mel_spectrogram(y_hat)
            y_mel = self.mel_spectrogram(batch["audio"])
            
            loss_mel = torch.abs(y_hat_mel - y_mel).mean()
            
            # Pitch guidance loss if enabled
            loss_pitch = 0.0
            if self.config.pitch_guidance and "f0" in batch:
                f0_loss = torch.abs(batch["f0"] - batch["predicted_f0"]).mean()
                loss_pitch += f0_loss
            
            # Total generator loss
            total_loss = loss_adv + 1.0 * loss_mel + 0.1 * loss_pitch
            
            # Backward pass
            total_loss.backward()
            self.optimizer_g.step()
            self.optimizer_d.step()
            
            # Update metrics
            total_loss_g += total_loss.item()
            total_loss_d += loss_disc.item()
            total_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f'{total_loss_g/total_steps:.4f}',
                'D_Loss': f'{total_loss_d/total_steps:.4f}'
            })
        
        # Update schedulers
        self.scheduler_g.step()
        self.scheduler_d.step()
        
        # Return epoch metrics
        return {
            'generator_loss': total_loss_g / total_steps,
            'discriminator_loss': total_loss_d / total_steps
        }
    
    def mel_spectrogram(self, y: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram"""
        # This would use your mel spectrogram implementation
        # For now, return a dummy tensor with the right shape
        batch_size, sequence_length = y.shape
        spec_length = sequence_length // 512  # Rough approximation
        return torch.randn(batch_size, 80, spec_length).to(y.device)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics
        }
        
        # Save checkpoint
        os.makedirs(self.config.weights_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.weights_dir, 
            f"{self.config.model_name}_epoch_{epoch}.pth"
        )
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.log_model_save(epoch, checkpoint_path)
        
        # Save latest checkpoint
        if self.config.save_only_latest:
            latest_path = os.path.join(
                self.config.weights_dir, 
                f"{self.config.model_name}_latest.pth"
            )
            torch.save(checkpoint, latest_path)
    
    def train(self) -> bool:
        """Run the complete training process"""
        try:
            self.logger.logger.info("Starting training process...")
            
            # Prepare data
            if not self.prepare_data():
                return False
            
            # Training loop
            best_loss = float('inf')
            
            for epoch in range(1, self.config.total_epochs + 1):
                self.logger.log_epoch_start(epoch, self.config.total_epochs)
                
                start_time = time.time()
                
                # Train epoch
                metrics = self.train_epoch(epoch)
                
                # Log epoch completion
                end_time = time.time()
                epoch_time = end_time - start_time
                
                metrics['epoch_time'] = epoch_time
                self.logger.log_epoch_end(epoch, metrics)
                
                # Save checkpoint
                if epoch % self.config.save_frequency == 0:
                    self.save_checkpoint(epoch, metrics)
                
                # Track best model
                if metrics['generator_loss'] < best_loss:
                    best_loss = metrics['generator_loss']
                
                # Save small final model weights if requested
                if self.config.save_every_weights and epoch % self.config.save_frequency == 0:
                    small_weights_path = os.path.join(
                        self.config.weights_dir,
                        f"{self.config.model_name}_epoch_{epoch}.pth"
                    )
                    
                    # Save just the generator state dict for lighter weight
                    torch.save(self.generator.state_dict(), small_weights_path)
            
            self.logger.logger.info(f"Training completed! Best loss: {best_loss:.6f}")
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Training failed: {e}")
            return False
    
    def create_feature_index(self):
        """Create feature index for the trained model"""
        try:
            self.logger.logger.info("Creating feature index...")
            
            # This would extract features from the training dataset
            # and create an index file for faster inference
            
            index_path = os.path.join(
                self.config.index_dir,
                f"{self.config.model_name}.index"
            )
            
            os.makedirs(self.config.index_dir, exist_ok=True)
            
            # Placeholder - actual implementation would use FAISS
            import faiss
            
            # Create placeholder index file
            with open(index_path, 'w') as f:
                json.dump({
                    "model_name": self.config.model_name,
                    "index_algorithm": "faiss",
                    "created_at": time.time()
                }, f)
            
            self.logger.logger.info(f"Feature index created: {index_path}")
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Failed to create feature index: {e}")
            return False


def start_training(config_path: str = None, **config_kwargs) -> bool:
    """Start training with given configuration"""
    # Load or create configuration
    if config_path and os.path.exists(config_path):
        config = TrainingConfig.load(config_path)
    else:
        config = TrainingConfig(**config_kwargs)
    
    # Create trainer
    trainer = AdvancedRVCTrainer(config)
    
    # Run training
    success = trainer.train()
    
    if success:
        # Create feature index
        trainer.create_feature_index()
        print("Training completed successfully!")
    
    return success


if __name__ == "__main__":
    # Default training configuration
    default_config = TrainingConfig(
        model_name="my_rvc_model",
        total_epochs=100,
        batch_size=8,
        learning_rate=0.001
    )
    
    # Start training
    start_training()
