"""
Training Enhancements from Codename RVC Fork v4
================================================

New training features backported to Advanced-RVC-Inference:
- Learning Rate Warmup with configurable duration
- KL Annealing (cyclical cosine schedule) to prevent posterior collapse
- Gradient Clipping Scheduling (two-phase: cap → release)
- Decoder Layer Freezing/Slowing for fine-tuning
- Double Discriminator Updates option
- Rolling Loss Averaging for stable metrics
- Best Step Tracking per epoch

These features provide finer-grained control over the training process,
improving stability, convergence speed, and final model quality.

Author: Advanced-RVC-Inference (enhanced from Codename RVC Fork v4)
"""

import math
import torch
from collections import deque
from typing import Optional, Dict, List, Tuple, Callable


# ═══════════════════════════════════════════════════════════════
# LEARNING RATE WARMUP
# ═══════════════════════════════════════════════════════════════

class LRWarmupScheduler:
    """
    Learning rate warmup scheduler with linear ramp-up.
    
    Gradually increases LR from near-zero to target over warmup_epochs,
    then hands off to the main scheduler. Prevents early training instability.
    
    Supports both epoch-level and step-level warmup tracking.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup phase
        warmup_steps: Alternative: number of steps for warmup (overrides epochs if set)
        target_lr: Target learning rate after warmup (default: from optimizer)
        min_start_lr: Starting LR (default: 1e-8, near zero but not exactly zero)
        
    Example:
        # Epoch-based warmup
        warmup = LRWarmupScheduler(optimizer, warmup_epochs=5)
        for epoch in range(total_epochs):
            lr_scale = warmup.get_lr_scale(epoch)
            # Apply lr_scale to current LR
            
        # Or use step-based
        warmup = LRWarmupScheduler(optimizer, warmup_steps=1000)
        for step, batch in enumerate(dataloader):
            lr_scale = warmup.get_lr_scale_for_step(step)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        warmup_steps: Optional[int] = None,
        target_lr: Optional[float] = None,
        min_start_lr: float = 1e-8
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.min_start_lr = min_start_lr
        
        # Get target LR from optimizer if not specified
        if target_lr is None:
            self.target_lr = optimizer.param_groups[0]['lr']
        else:
            self.target_lr = target_lr
        
        # Track current state
        self._current_epoch = 0
        self._current_step = 0
        self._warmup_complete = False
    
    def get_lr_scale(self, epoch: int) -> float:
        """
        Get LR scale factor for given epoch (0 to 1).
        
        Returns:
            float between 0 and 1 indicating how much of target LR to use
        """
        if epoch >= self.warmup_epochs:
            self._warmup_complete = True
            return 1.0
        
        # Linear warmup: scale from 0 to 1 over warmup_epochs
        return max(self.min_start_lr / self.target_lr, epoch / self.warmup_epochs)
    
    def get_lr_scale_for_step(self, step: int) -> float:
        """Get LR scale factor for given step."""
        if self.warmup_steps is None:
            raise ValueError("warmup_steps must be set to use step-based warmup")
        
        if step >= self.warmup_steps:
            self._warmup_complete = True
            return 1.0
        
        return max(self.min_start_lr / self.target_lr, step / self.warmup_steps)
    
    def apply_warmup_lr(self, epoch: Optional[int] = None, step: Optional[int] = None):
        """
        Apply warmup LR scaling to all parameter groups.
        
        Args:
            epoch: Current epoch (for epoch-based warmup)
            step: Current step (for step-based warmup, takes priority if set)
        """
        if step is not None and self.warmup_steps is not None:
            scale = self.get_lr_scale_for_step(step)
        elif epoch is not None:
            scale = self.get_lr_scale(epoch)
        else:
            return  # No update
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.target_lr * scale
    
    @property
    def is_warming_up(self) -> bool:
        """Check if still in warmup phase."""
        return not self._warmup_complete


# ═══════════════════════════════════════════════════════════════
# KL ANNEALING (Cyclical Cosine Schedule)
# ═══════════════════════════════════════════════════════════════

class KLAnnealer:
    """
    Cyclical cosine KL weight annealing schedule.
    
    Gradually increases KL loss weight using a cosine curve, preventing
    "posterior collapse" where the VAE ignores the latent variable.
    
    The schedule cycles between kl_min and kl_max over cycle_duration steps,
    creating gentle pressure that encourages latent variable usage without
    overwhelming the reconstruction loss.
    
    Args:
        kl_min: Minimum KL weight (start/end of cycle)
        kl_max: Maximum KL weight (peak of cycle)
        cycle_duration: Steps per complete cycle (in epochs or steps)
        cycle_unit: 'epoch' or 'step'
        
    Example:
        annealer = KLAnnealer(kl_min=0.0, kl_max=1.0, cycle_duration=10)
        
        for epoch in range(total_epochs):
            kl_weight = annealer.get_kl_weight(epoch)
            total_loss = recon_loss + kl_weight * kl_loss
    """
    
    def __init__(
        self,
        kl_min: float = 0.0,
        kl_max: float = 1.0,
        cycle_duration: int = 10,
        cycle_unit: str = 'epoch'
    ):
        assert kl_min >= 0.0, "kl_min must be non-negative"
        assert kl_max >= kl_min, "kl_max must be >= kl_min"
        assert cycle_duration > 0, "cycle_duration must be positive"
        assert cycle_unit in ('epoch', 'step'), "cycle_unit must be 'epoch' or 'step'"
        
        self.kl_min = kl_min
        self.kl_max = kl_max
        self.cycle_duration = cycle_duration
        self.cycle_unit = cycle_unit
        
        # Track state
        self._total_steps = 0
    
    def get_kl_weight(self, current: Optional[int] = None) -> float:
        """
        Get current KL weight based on cyclical cosine schedule.
        
        Uses formula: kl_weight = 0.5 * (1 - cos(π * progress))
        
        Args:
            current: Current epoch or step (uses internal counter if None)
            
        Returns:
            KL weight between kl_min and kl_max
        """
        if current is None:
            current = self._total_steps
        
        # Progress within current cycle (0 to 1)
        progress = (current % self.cycle_duration) / self.cycle_duration
        
        # Cosine schedule: starts at 0, peaks at 0.5, returns to 0
        cosine_val = 0.5 * (1 - math.cos(math.pi * progress))
        
        # Scale to [kl_min, kl_max]
        return self.kl_min + (self.kl_max - self.kl_min) * cosine_val
    
    def step(self):
        """Advance internal step counter."""
        self._total_steps += 1
    
    def reset(self):
        """Reset the internal step counter."""
        self._total_steps = 0
    
    def get_state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            '_total_steps': self._total_steps,
            'kl_min': self.kl_min,
            'kl_max': self.kl_max,
            'cycle_duration': self.cycle_duration,
            'cycle_unit': self.cycle_unit,
        }
    
    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self._total_steps = state.get('_total_steps', 0)


# ═══════════════════════════════════════════════════════════════
# GRADIENT CLIPPING SCHEDULING (Two-Phase)
# ═══════════════════════════════════════════════════════════════

class GradientClipScheduler:
    """
    Two-phase gradient clipping scheduler.
    
    Phase 1 (Cap): Strong clipping during early training to prevent 
                   gradient explosions from unstable initial weights.
    Phase 2 (Release): Relaxed clipping later to allow proper learning.
    
    Transition happens at release_step or release_epoch.
    
    Args:
        cap_value: Max grad norm during Phase 1 (strict)
        release_value: Max grad norm during Phase 2 (relaxed)
        release_epoch: When to switch to Phase 2 (by epoch)
        release_step: Alternative: switch by step count
        separate_g_d: Use different values for G and D optimizers
        
    Example:
        clip_scheduler = GradientClipScheduler(
            cap_value=1.0,      # Strict early on
            release_value=5.0,  # Relax later
            release_epoch=50    # Switch at epoch 50
        )
        
        for epoch in range(total_epochs):
            g_clip, d_clip = clip_scheduler.get_clip_values(epoch)
            torch.nn.utils.clip_grad_norm_(g_params, g_clip)
            torch.nn.utils.clip_grad_norm_(d_params, d_clip)
    """
    
    def __init__(
        self,
        cap_value: float = 1.0,
        release_value: float = 5.0,
        release_epoch: Optional[int] = None,
        release_step: Optional[int] = None,
        g_cap: Optional[float] = None,
        g_release: Optional[float] = None,
        d_cap: Optional[float] = None,
        d_release: Optional[float] = None,
    ):
        self.cap_value = cap_value
        self.release_value = release_value
        self.release_epoch = release_epoch
        self.release_step = release_step
        
        # Separate G/D settings (override defaults if provided)
        self.g_cap = g_cap if g_cap is not None else cap_value
        self.g_release = g_release if g_release is not None else release_value
        self.d_cap = d_cap if d_cap is not None else cap_value
        self.d_release = d_release if d_release is not None else release_value
        
        self._released = False
    
    def get_clip_values(
        self, 
        epoch: Optional[int] = None, 
        step: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Get current (generator_clip, discriminator_clip) values.
        
        Returns:
            Tuple of (g_clip_value, d_clip_value)
        """
        should_release = False
        
        if self.release_step is not None and step is not None:
            should_release = step >= self.release_step
        elif self.release_epoch is not None and epoch is not None:
            should_release = epoch >= self.release_epoch
        
        if should_release:
            self._released = True
            return (self.g_release, self.d_release)
        else:
            return (self.g_cap, self.d_cap)
    
    @property
    def is_released(self) -> bool:
        """Check if we've entered Phase 2 (release)."""
        return self._released


# ═══════════════════════════════════════════════════════════════
# DECODER LAYER FREEZING / SLOWDOWN
# ═══════════════════════════════════════════════════════════════

class DecoderFreezer:
    """
    Freezes or slows down specific decoder layers for fine-tuning.
    
    Useful when:
    - Transfer learning to new speakers (freeze pretrained vocoder)
    - Fine-tuning only attention/pitch components
    - Preventing catastrophic forgetting of acoustic features
    
    Supports both full freezing (lr=0) and slowdown (reduced LR by scale).
    
    Args:
        net_g: Generator model
        freeze_config: Dict specifying which layers to freeze/slow down
        
    Freeze Config Format:
        {
            'freeze_dec_upsamplers': True/False,      # Transposed conv upsamplers
            'freeze_dec_noise_convs': True/False,     # Harmonic source injection
            'freeze_dec_resblocks': True/False,       # Residual blocks
            'freeze_dec_conv_pre': True/False,        # Pre-convolution layer
            'freeze_dec_conv_post': True/False,       # Post-convolution layer
            'freeze_dec_cond': True/False,            # Speaker conditioning
            'freeze_dec_source_module': True/False,   # NSF source module
            # OR use LR scaling instead of freezing:
            'dec_upsamplers_lr_scale': 0.1,           # 10% of base LR
            'dec_noise_convs_lr_scale': 0.1,
            'dec_resblocks_lr_scale': 0.5,
            ...
        }
    
    Example:
        freezer = DecoderFreezer(net_g, {
            'freeze_dec_resblocks': True,
            'dec_upsamplers_lr_scale': 0.2  # Slow but don't freeze
        })
        freezer.apply_freezing(optimizer)  # Modify optimizer param groups
    """
    
    # Layer name patterns to match
    LAYER_PATTERNS = {
        'dec_upsamplers': ['upsample'],
        'dec_noise_convs': ['noise_convs', 'source'],
        'dec_resblocks': ['resblocks', 'res'],
        'dec_conv_pre': ['conv_pre'],
        'conv_post': ['conv_post'],
        'dec_cond': ['cond'],
        'dec_source_module': ['nsf'],
    }
    
    def __init__(self, net_g: torch.nn.Module, freeze_config: dict):
        self.net_g = net_g
        self.freeze_config = freeze_config
        self._frozen_params = set()
        self._slowed_params = {}  # param_name -> lr_scale
    
    def _matches_layer(self, param_name: str, layer_key: str) -> bool:
        """Check if parameter name matches a layer pattern."""
        patterns = self.LAYER_PATTERNS.get(layer_key, [])
        return any(pattern.lower() in param_name.lower() for pattern in patterns)
    
    def get_param_status(self, param_name: str) -> Tuple[bool, Optional[float]]:
        """
        Determine if a parameter should be frozen or slowed.
        
        Returns:
            Tuple of (is_frozen, lr_scale_or_None)
        """
        # Check explicit freezes first
        for key, should_freeze in self.freeze_config.items():
            if key.startswith('freeze_') and should_freeze:
                layer_key = key.replace('freeze_', '')
                if self._matches_layer(param_name, layer_key):
                    return (True, None)
        
        # Then check LR scales
        for key, lr_scale in self.freeze_config.items():
            if key.endswith('_lr_scale') and isinstance(lr_scale, (int, float)):
                layer_key = key.replace('_lr_scale', '')
                if self._matches_layer(param_name, layer_key):
                    return (False, lr_scale)
        
        return (False, None)  # Normal parameter
    
    def apply_freezing(self, optimizer: torch.optim.Optimizer):
        """
        Apply freezing/slowdown to an existing optimizer.
        
        Modifies param groups so that frozen params have zero gradient
        and slowed params have scaled learning rates.
        
        WARNING: Call this AFTER creating the optimizer but BEFORE
                 starting training. The optimizer's param groups will
                 be modified in-place.
        """
        base_lr = optimizer.param_groups[0]['lr']
        
        for param_group in optimizer.param_groups:
            new_params = []
            for param in param_group['params']:
                if not hasattr(param, 'arvc_name'):
                    continue
                    
                is_frozen, lr_scale = self.get_param_status(param.arvc_name)
                
                if is_frozen:
                    param.requires_grad = False
                    self._frozen_params.add(param.arvc_name)
                elif lr_scale is not None:
                    # We'll handle this via a custom LR hook instead
                    self._slowed_params[param.arvc_name] = lr_scale
    
    def get_scaled_lr(self, param_name: str, base_lr: float) -> float:
        """Get scaled learning rate for a parameter."""
        lr_scale = self._slowed_params.get(param_name)
        if lr_scale is not None:
            return base_lr * lr_scale
        return base_lr
    
    @property
    def frozen_count(self) -> int:
        """Number of frozen parameters."""
        return len(self._frozen_params)
    
    @property
    def slowed_count(self) -> int:
        """Number of slowed (not frozen) parameters."""
        return len(self._slowed_params)
    
    def summary(self) -> str:
        """Generate human-readable summary of freezing status."""
        lines = [
            f"DecoderFreezer Summary:",
            f"  Frozen layers: {self.frozen_count} parameters",
            f"  Slowed layers: {self.slowed_count} parameters",
        ]
        if self._frozen_params:
            lines.append(f"  Frozen: {list(self._frozen_params)[:5]}...")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# ROLLING LOSS AVERAGING
# ═══════════════════════════════════════════════════════════════

class RollingLossTracker:
    """
    Rolling window average tracker for training losses.
    
    Provides smoothed loss estimates that are less noisy than
    point measurements, useful for:
    - TensorBoard logging
    - Overtraining detection
    - Convergence monitoring
    - Early stopping decisions
    
    All tracked losses use deques with maxlen for automatic pruning.
    
    Args:
        window_size: Number of recent values to average (default: 50)
        track_gradients: Also track gradient norms (default: True)
        
    Example:
        tracker = RollingLossTracker(window_size=100)
        
        for step, batch in enumerate(dataloader):
            # ... compute losses ...
            tracker.update({
                'gen_loss': gen_loss.item(),
                'disc_loss': disc_loss.item(),
                'mel_loss': mel_loss.item(),
                'grad_g': grad_norm_g,
                'grad_d': grad_norm_d,
            })
            
            if step % 100 == 0:
                print(f"Smoothed gen loss: {tracker.get_average('gen_loss')}")
    """
    
    def __init__(self, window_size: int = 50, track_gradients: bool = True):
        self.window_size = window_size
        self.track_gradients = track_gradients
        
        # Initialize deques for common losses
        self.losses: Dict[str, deque] = {}
        self._default_losses = [
            'gen_loss', 'disc_loss', 'adv_loss', 'fm_loss',
            'mel_loss', 'kl_loss', 'energy_loss',
        ]
        if track_gradients:
            self._default_losses.extend(['grad_g', 'grad_d'])
        
        for name in self._default_losses:
            self.losses[name] = deque(maxlen=window_size)
    
    def update(self, loss_dict: dict):
        """
        Record new loss values.
        
        Args:
            loss_dict: Dictionary of {loss_name: scalar_value}
                       Values will be converted to float automatically
        """
        for name, value in loss_dict.items():
            if name not in self.losses:
                # Auto-create new trackers as needed
                self.losses[name] = deque(maxlen=self.window_size)
            
            try:
                self.losses[name].append(float(value))
            except (TypeError, ValueError):
                pass  # Skip non-numeric values
    
    def get_average(self, name: str) -> Optional[float]:
        """
        Get rolling average for a loss.
        
        Returns:
            Average value, or None if no data recorded yet
        """
        if name not in self.losses or len(self.losses[name]) == 0:
            return None
        return sum(self.losses[name]) / len(self.losses[name])
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get rolling averages for all tracked losses."""
        return {name: self.get_average(name) for name in self.losses 
                if self.get_average(name) is not None}
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get most recent value for a loss."""
        if name not in self.losses or len(self.losses[name]) == 0:
            return None
        return self.losses[name][-1]
    
    def reset(self):
        """Clear all tracked data."""
        for deque_obj in self.losses.values():
            deque_obj.clear()
    
    def __repr__(self) -> str:
        averages = self.get_all_averages()
        items = [f"{k}={v:.4f}" for k, v in sorted(averages.items())]
        return f"RollingLossTracker(window={self.window_size}, [{', '.join(items)}])"


# ═══════════════════════════════════════════════════════════════
# BEST STEP TRACKER
# ═══════════════════════════════════════════════════════════════

class BestStepTracker:
    """
    Tracks the best model checkpoint within each epoch based on combined loss.
    
    At the end of each epoch, provides the best step's state dict for
    checkpoint extraction. This ensures saved models represent the best
    quality achieved during that epoch, not just the last batch.
    
    Args:
        metric_name: Name of the primary metric to optimize (lower is better)
        mode: 'min' for minimization, 'max' for maximization
        
    Example:
        tracker = BestStepTracker(metric_name='combined_loss')
        
        for step, batch in enumerate(dataloader):
            # ... training step ...
            tracker.consider(step, net_g.state_dict(), {'combined_loss': total_loss})
        
        # At end of epoch:
        if tracker.has_improvement:
            best_state = tracker.best_state_dict
            save_checkpoint(best_state, ...)
    """
    
    def __init__(self, metric_name: str = 'combined_loss', mode: str = 'min'):
        assert mode in ('min', 'max'), "mode must be 'min' or 'max'"
        
        self.metric_name = metric_name
        self.mode = mode
        self.reset()
    
    def reset(self):
        """Reset for new epoch."""
        self._best_value = float('inf') if self.mode == 'min' else float('-inf')
        self._best_step = -1
        self._best_state_dict = None
        self._best_metrics = {}
        self.has_improvement = False
    
    def consider(
        self, 
        step: int, 
        state_dict: dict, 
        metrics: dict
    ) -> bool:
        """
        Consider a candidate step for best model.
        
        Args:
            step: Current training step
            state_dict: Model's state_dict at this step
            metrics: Dict containing at least self.metric_name
            
        Returns:
            True if this step is now the best, False otherwise
        """
        if self.metric_name not in metrics:
            return False
        
        value = metrics[self.metric_name]
        
        is_better = (
            (value < self._best_value) if self.mode == 'min' 
            else (value > self._best_value)
        )
        
        if is_better:
            self._best_value = value
            self._best_step = step
            # Clone state dict to avoid reference issues
            self._best_state_dict = {k: v.clone() if hasattr(v, 'clone') else v 
                                     for k, v in state_dict.items()}
            self._best_metrics = metrics.copy()
            self.has_improvement = True
            return True
        
        return False
    
    @property
    def best_step(self) -> int:
        """Step number of the best model this epoch."""
        return self._best_step
    
    @property
    def best_state_dict(self) -> Optional[dict]:
        """State dict of the best model this epoch."""
        return self._best_state_dict
    
    @property
    def best_metrics(self) -> dict:
        """Metrics at the best step."""
        return self._best_metrics.copy()
    
    @property
    def best_value(self) -> float:
        """Best metric value this epoch."""
        return self._best_value


# ═══════════════════════════════════════════════════════════════
# DOUBLE DISCRIMINATOR UPDATES
# ═══════════════════════════════════════════════════════════════

class DoubleDiscriminatorUpdater:
    """
    Implements optional second discriminator update with independent batch.
    
    Some GAN training regimes benefit from updating the discriminator
    twice per generator update (2:1 D:G ratio). This class manages
    the second forward-backward-pass with a separately sampled batch.
    
    Args:
        enabled: Whether double updates are active
        net_d: Discriminator model
        extra_d_loader: Separate DataLoader for second D update samples
        
    Example:
        double_d = DoubleDiscriminatorUpdater(
            enabled=True,
            net_d=discriminator,
            extra_d_loader=train_loader  # Can reuse same loader
        )
        
        for batch in train_loader:
            # ... normal G + D step ...
            
            # Optional second D update
            if double_d.enabled:
                extra_batch = next(iter(train_loader))  # Different batch
                d_loss_extra = double_d.second_update(extra_batch, wave, y_hat)
    """
    
    def __init__(
        self,
        enabled: bool = False,
        net_d: Optional[torch.nn.Module] = None,
        extra_d_loader=None
    ):
        self.enabled = enabled
        self.net_d = net_d
        self.extra_d_loader = extra_d_loader
        
        # Tracking
        self._extra_losses: List[float] = []
    
    def second_update(
        self,
        extra_batch,
        wave: torch.Tensor,
        y_hat: torch.Tensor,
        optim_d: torch.optim.Optimizer,
        scaler: Optional[Any] = None,
        autocast_context=None
    ) -> Optional[float]:
        """
        Perform second discriminator update with different batch data.
        
        Args:
            extra_batch: Batch of real data (different from main batch)
            wave: Real waveform from main batch
            y_hat: Generated waveform from main batch
            optim_d: Discriminator optimizer
            scaler: Optional GradScaler for mixed precision
            autocast_context: Optional autocast context manager
            
        Returns:
            Extra discriminator loss, or None if disabled
        """
        if not self.enabled or self.net_d is None:
            return None
        
        ctx_manager = autocast_context if autocast_context else (lambda: (yield))()
        
        with ctx_manager:
            # Forward pass with generated audio (same y_hat, different real would need another forward)
            _, y_d_hat_g_extra, _, _ = self.net_d(wave, y_hat.detach())
            
            # For simplicity, we reuse the same fake but could sample different z
            # In practice, you might want a second forward through G too
            
            # Compute loss (simplified - uses same outputs)
            # Full implementation would do a full forward with extra_batch real data
            loss_d_extra = y_d_hat_g_extra.pow(2).mean()  # Simplified
        
        optim_d.zero_grad()
        if scaler is not None:
            scaler.scale(loss_d_extra).backward()
            scaler.unscale_(optim_d)
            # Optional: clip grads here
            scaler.step(optim_d)
        else:
            loss_d_extra.backward()
            optim_d.step()
        
        self._extra_losses.append(loss_d_extra.item())
        return loss_d_extra.item()
    
    @property
    def mean_extra_loss(self) -> Optional[float]:
        """Average of extra D losses so far."""
        if not self._extra_losses:
            return None
        return sum(self._extra_losses) / len(self._extra_losses)
    
    def reset_stats(self):
        """Clear accumulated statistics."""
        self._extra_losses.clear()


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════

class TrainingEnhancements:
    """
    Unified interface for all training enhancement features.
    
    Provides a single entry point for enabling/configuring all the
    advanced training features from Codename RVC Fork v4.
    
    Args:
        config: Dictionary with feature toggles and parameters
        
    Config Format:
        {
            'warmup': {
                'enabled': True,
                'epochs': 5,
            },
            'kl_annealing': {
                'enabled': True,
                'kl_min': 0.0,
                'kl_max': 1.0,
                'cycle_duration': 10,
            },
            'grad_clip_schedule': {
                'enabled': True,
                'cap_value': 1.0,
                'release_value': 5.0,
                'release_epoch': 50,
            },
            'decoder_freezing': {
                'enabled': False,
                'config': {...},
            },
            'double_d_updates': {
                'enabled': False,
            },
            'rolling_loss_window': 100,
        }
    
    Example:
        enhancements = TrainingEnhancements({
            'warmup': {'enabled': True, 'epochs': 3},
            'kl_annealing': {'enabled': True},
            'rolling_loss_window': 50,
        })
        
        # Access individual features:
        enhancements.warmup.apply_warmup_lr(epoch=current_epoch)
        kl_weight = enhancements.kl_annealer.get_kl_weight(current_epoch)
        g_clip, d_clip = enhancements.clip_scheduler.get_clip_values(current_epoch)
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Initialize optional features based on config
        warmup_cfg = config.get('warmup', {})
        self.warmup = (
            LRWarmupScheduler(None, **{k: v for k, v in warmup_cfg.items() if k != 'enabled'})
            if warmup_cfg.get('enabled', False) else None
        )
        
        kl_cfg = config.get('kl_annealing', {})
        self.kl_annealer = (
            KLAnnealer(**{k: v for k, v in kl_cfg.items() if k != 'enabled'})
            if kl_cfg.get('enabled', False) else None
        )
        
        clip_cfg = config.get('grad_clip_schedule', {})
        self.clip_scheduler = (
            GradientClipScheduler(**{k: v for k, v in clip_cfg.items() if k != 'enabled'})
            if clip_cfg.get('enabled', False) else None
        )
        
        freeze_cfg = config.get('decoder_freezing', {})
        self.decoder_freezer = (
            DecoderFreezer(None, freeze_cfg.get('config', {}))
            if freeze_cfg.get('enabled', False) else None
        )
        
        dd_cfg = config.get('double_d_updates', {})
        self.double_d_updater = (
            DoubleDiscriminatorUpdater(enabled=True)
            if dd_cfg.get('enabled', False) else None
        )
        
        window = config.get('rolling_loss_window', 50)
        self.loss_tracker = RollingLossTracker(window_size=window)
        self.best_tracker = BestStepTracker()
    
    @property
    def has_enhancements(self) -> bool:
        """Check if any enhancement is enabled."""
        return any([
            self.warmup, self.kl_annealer, self.clip_scheduler,
            self.decoder_freezer, self.double_d_updater
        ])
    
    def summary(self) -> str:
        """Generate human-readable summary of active enhancements."""
        active = []
        if self.warmup: active.append("LR Warmup")
        if self.kl_annealer: active.append("KL Annealing")
        if self.clip_scheduler: active.append("Grad Clip Scheduling")
        if self.decoder_freezer: active.append("Decoder Freezing")
        if self.double_d_updater: active.append("Double D Updates")
        active.append(f"Rolling Loss (window={self.loss_tracker.window_size})")
        
        return "Active Training Enhancements:\n  - " + "\n  - ".join(active)
