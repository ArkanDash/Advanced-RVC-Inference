"""
Enhanced Loss Functions for RVC Training
=========================================

Backported from Codename RVC Fork v4 with additional improvements:
- Phase loss for RingFormer architectures (ISTFT-based vocoders)
- Envelope loss for waveform envelope matching  
- Free-bits KL loss to prevent posterior collapse
- MultiScaleSTFTLoss for enhanced spectral fidelity
- Original losses with bug fixes from Vietnamese-RVC/Applio

Author: Advanced-RVC-Inference (enhanced from Codename RVC Fork v4)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# ORIGINAL LOSSES (with Vietnamese-RVC bug fixes)
# ═══════════════════════════════════════════════════════════════

def feature_loss(fmap_r, fmap_g):
    """
    Compute the feature loss between reference and generated feature maps.
    
    BUG FIX (Vietnamese-RVC): .detach() on real features prevents building
    autograd graph for real path during G step — pure waste since G optimizer
    never updates D params.
    
    Args:
        fmap_r (list of torch.Tensor): List of reference feature maps.
        fmap_g (list of torch.Tensor): List of generated feature maps.
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            # Detach real features (from Vietnamese-RVC): during the G step,
            # fmap_r comes from a real-wave forward pass through D. Without
            # .detach(), the autograd graph is built for the real path even
            # though the G optimizer never updates D params — pure waste.
            loss += (rl.float().detach() - gl.float()).abs().mean()

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    LSGAN-style discriminator loss.
    
    Returns:
        Tuple of (total_loss, list_of_real_losses, list_of_fake_losses)
    """
    loss = 0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = ((1 - dr) ** 2).mean()
        g_loss = (dg**2).mean()
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """
    LSGAN Generator Loss.
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = ((1 - dg.float()) ** 2).mean()
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def discriminator_loss_scaled(disc_real, disc_fake, scale=1.0):
    """Scaled discriminator loss (from Applio).

    Downweights losses from sub-discriminators beyond the midpoint by `scale`.
    This prevents multi-resolution sub-discriminator heads (v3) from dominating
    the total loss, which can improve training stability and quality.
    """
    midpoint = len(disc_real) // 2
    r_losses, g_losses = [], []
    loss = 0
    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        real_loss = (1 - d_real.float()).pow(2).mean()
        fake_loss = d_fake.float().pow(2).mean()
        total_loss = real_loss + fake_loss
        if i >= midpoint:
            total_loss = total_loss * scale
            real_loss = real_loss * scale
            fake_loss = fake_loss * scale
        loss += total_loss
        r_losses.append(real_loss.item())
        g_losses.append(fake_loss.item())
    return loss, r_losses, g_losses


def generator_loss_scaled(disc_outputs, scale=1.0):
    """Scaled generator loss (from Applio).

    Downweights losses from sub-discriminators beyond the midpoint by `scale`.
    This prevents multi-resolution sub-discriminator heads (v3) from dominating
    the total loss, which can improve training stability and quality.
    """
    midpoint = len(disc_outputs) // 2
    gen_losses = []
    loss = 0
    for i, d_fake in enumerate(disc_outputs):
        loss_value = (1 - d_fake.float()).pow(2).mean()
        if i >= midpoint:
            loss_value = loss_value * scale
        gen_losses.append(loss_value)
        loss += loss_value
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Standard KL divergence loss for VAE regularization.
    
    Args:
        z_p: Sampled latent variable transformed by the flow [b, h, t_t]
        logs_q: Log variance of the posterior distribution q [b, h, t_t]
        m_p: Mean of the prior distribution p [b, h, t_t]
        logs_p: Log variance of the prior distribution p [b, h, t_t]
        z_mask: Mask for the latent variables [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * (-2.0 * logs_p).exp()

    return (kl * z_mask).sum() / z_mask.sum()


# ═══════════════════════════════════════════════════════════════
# NEW LOSSES FROM CODENAME RVC FORK v4
# ═══════════════════════════════════════════════════════════════

def phase_loss(x_fft: torch.Tensor, g_fft: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Phase coherence loss for RingFormer ISTFT-based vocoders.
    
    Measures cosine similarity between normalized STFT phases of target
    and generated audio. Critical for RingFormer v1/v2 architectures
    which use ISTFT output (phase-aware) vs traditional HiFi-GAN
    (phase-insensitive).
    
    Args:
        x_fft: Complex STFT of target audio
        g_fft: Complex STFT of generated audio  
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        Phase coherence loss (lower is better, 0 = perfect phase match)
    
    Usage:
        # In training loop for RingFormer vocoders:
        x_stft = torch.stft(wave, n_fft, hop, win, return_complex=True)
        g_stft = torch.stft(y_hat, n_fft, hop, win, return_complex=True)
        loss_phase = phase_loss(x_stft, g_stft) * lambda_phase
    """
    x_norm = x_fft / (x_fft.abs() + 1e-9)
    g_norm = g_fft / (g_fft.abs() + 1e-9)

    phase_similarity = (x_norm * g_norm.conj()).real
    loss = 1.0 - phase_similarity

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unsupported reduction mode: {reduction}")


def envelope_loss(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    """
    Waveform envelope matching loss.
    
    Uses MaxPool1d to extract positive (peaks) and negative (troughs) 
    envelopes, then computes L1 distance between them. This captures
    the overall amplitude shape without being sensitive to phase.
    
    Particularly useful for:
    - Preventing volume mismatches between source and target
    - Improving perceived loudness consistency
    - Complementing spectral losses that miss temporal envelope
    
    Args:
        y: Target waveform [B, 1, T]
        y_hat: Generated waveform [B, 1, T]
        
    Returns:
        Combined L1 loss on positive + negative envelopes
        
    Note:
        stride < kernel_size ensures overlapping coverage so no 
        peaks/troughs are missed at pool boundaries
    """
    # stride < kernel_size ensures overlapping coverage so no spikes are missed
    m = torch.nn.MaxPool1d(kernel_size=5, stride=3)

    # Positive envelope (peaks)
    y_env = m(y)
    y_hat_env = m(y_hat)

    # Negative envelope (troughs)
    y_rev_env = m(-y)
    y_hat_rev_env = m(-y_hat)

    return torch.nn.functional.l1_loss(y_env, y_hat_env) + \
           torch.nn.functional.l1_loss(y_rev_env, y_hat_rev_env)


def kl_loss_fb(
    z_p: torch.Tensor, 
    logs_q: torch.Tensor, 
    m_p: torch.Tensor, 
    logs_p: torch.Tensor, 
    z_mask: torch.Tensor, 
    z_p2: Optional[torch.Tensor] = None, 
    free_bits: float = 0.0
) -> torch.Tensor:
    """
    KL divergence loss with Free Bits mechanism (Kingma et al., 2016).
    
    Prevents "posterior collapse" where the VAE learns to ignore the latent
    variable by always producing near-zero KL. The free_bits floor ensures
    each dimension contributes at least some information.
    
    Key differences from standard kl_loss:
    1. Per-dimension KL floor (free_bits / n_dims per dimension)
    2. Optional 2-sample estimation for more robust KL estimates
    3. Better numerical stability for very small KL values
    
    Args:
        z_p: Sampled latent variable transformed by the flow [b, h, t_t]
        logs_q: Log variance of the posterior distribution q [b, h, t_t]
        m_p: Mean of the prior distribution p [b, h, t_t]
        logs_p: Log variance of the prior distribution p [b, h, t_t]
        z_mask: Mask for the latent variables [b, 1, t_t] or [b, h, t_t]
        z_p2: Optional second independent sample through flow (for 2-sample estimate)
        free_bits: Total KL floor in nats (divided across dims internally).
                   e.g., free_bits=1.0 with 192 dims -> 0.0052 nats/dim minimum.
                   Set to 0.0 to disable (equivalent to standard kl_loss).
                   
    Returns:
        Scalar KL loss with free-bits floor applied
        
    Example:
        # Standard usage (prevents collapse):
        loss_kl = kl_loss_fb(z_p, logs_q, m_p, logs_p, z_mask, free_bits=1.0)
        
        # With 2-sample estimation (more robust):
        loss_kl = kl_loss_fb(z_p, logs_q, m_p, logs_p, z_mask, z_p2=z_p2_2, free_bits=0.5)
    """
    def _term(zp):
        return logs_p - logs_q - 0.5 + 0.5 * ((zp - m_p) ** 2) * torch.exp(-2 * logs_p)

    if z_p2 is not None:
        kl = (_term(z_p) + _term(z_p2)) * 0.5
    else:
        kl = _term(z_p)

    # kl: [b, h, t_t], z_mask: [b, 1, t_t] or [b, h, t_t]
    kl = kl * z_mask

    # Per-dim KL: sum over batch and time, average over valid elements per dim
    # [b, h, t_t] -> [h]
    n_dims = z_p.size(1)
    kl_per_dim = kl.sum(dim=(0, 2))
    mask_per_dim = z_mask.sum(dim=(0, 2)).clamp(min=1)
    kl_per_dim = kl_per_dim / mask_per_dim

    # Apply free bits floor (total floor divided across dims)
    per_dim_floor = free_bits / n_dims
    kl_per_dim = kl_per_dim.clamp(min=per_dim_floor)

    # Sum over dims (matches old kl_loss scale: old divided by z_mask.sum()=b*t, not b*h*t)
    loss = kl_per_dim.sum()

    return loss


class MultiScaleSTFTLoss(nn.Module):
    """
    Multi-scale STFT loss for audio reconstruction quality.
    
    Computes spectral convergence AND log magnitude L1 loss at multiple
    STFT resolutions (fft/hop/window sizes). Captures both coarse 
    spectral structure (large fft) and fine details (small fft).
    
    Advantages over single-scale mel loss:
    1. Multi-resolution: catches artifacts at different time scales
    2. Spectral convergence: frequency-domain normalization
    3. Silence masking: excludes silent frames from SC (undefined)
    
    Typical configurations:
    - Default: (512, 1024, 2048) - good balance of speed/quality
    - High quality: (512, 1024, 2048, 4096) - slower but more accurate
    - Fast: (256, 512, 1024) - for quick experimentation
    
    Usage:
        stft_loss = MultiScaleSTFTLoss(
            fft_sizes=(512, 1024, 2048),
            hop_sizes=(128, 256, 512),
            win_sizes=(512, 1024, 2048)
        )
        loss = stft_loss(predicted_audio, target_audio)
    """

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Tuple[int, ...] = (128, 256, 512),
        win_sizes: Tuple[int, ...] = (512, 1024, 2048),
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes), \
            "fft_sizes, hop_sizes, and win_sizes must have the same length"

    def _stft(self, x: torch.Tensor, fft_size: int, hop_size: int, win_size: int) -> torch.Tensor:
        """Compute STFT magnitude spectrum.

        Args:
            x: Audio tensor [B, C, T] or [B, T]
            fft_size: FFT size
            hop_size: Hop length
            win_size: Window size
            
        Returns:
            Magnitude spectrogram [B, F, T']
        """
        # [B, C, T] -> [B, T]
        x = x.squeeze(1) if x.dim() == 3 else x

        # Pad to avoid edge effects
        x = F.pad(x, (win_size // 2, win_size // 2), mode='reflect')

        window = torch.hann_window(win_size, device=x.device, dtype=x.dtype)
        stft = torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True, center=False
        )
        return stft.abs()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale STFT loss.
        
        Per-sample spectral convergence with silence masking —
        mute samples (||X||_F ≈ 0) are excluded since SC is undefined for zero-energy.
        
        Args:
            pred: (B, T) or (B, 1, T) predicted audio
            target: (B, T) or (B, 1, T) target audio
            
        Returns:
            Combined spectral convergence + log magnitude loss
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            pred_mag = self._stft(pred, fft_size, hop_size, win_size)      # [B, F, T]
            target_mag = self._stft(target, fft_size, hop_size, win_size)  # [B, F, T]

            # Per-sample Frobenius norms
            flat_target = target_mag.reshape(target_mag.size(0), -1)           # [B, F*T]
            flat_diff = (target_mag - pred_mag).reshape(target_mag.size(0), -1)
            target_nrg = torch.norm(flat_target, p=2, dim=1)                # [B]
            diff_nrg = torch.norm(flat_diff, p=2, dim=1)                    # [B]

            # Mask out silent samples (SC is undefined for zero-energy)
            mask = target_nrg > 1e-4
            if mask.any():
                sc_loss += (diff_nrg[mask] / target_nrg[mask]).mean()

            # Log magnitude loss — safe for all samples (clamp avoids -inf)
            mag_loss += F.l1_loss(
                torch.log(pred_mag.clamp(min=1e-5)),
                torch.log(target_mag.clamp(min=1e-5)),
            )

        sc_loss = sc_loss / len(self.fft_sizes) if sc_loss != 0.0 else 0.0
        mag_loss = mag_loss / len(self.fft_sizes)
        return sc_loss + mag_loss


# ═══════════════════════════════════════════════════════════════
# COMBINED LOSS FUNCTION (convenience wrapper)
# ═══════════════════════════════════════════════════════════════

class EnhancedLossCalculator:
    """
    Unified loss calculator supporting all loss types from this module.
    
    Provides a clean interface for the training loop to compute all
    losses with proper weighting and conditional activation.
    
    Example:
        calculator = EnhancedLossCalculator(
            config=config,
            use_phase_loss=vocoder.startswith("RingFormer"),
            use_envelope_loss=True,
            use_free_bits_kl=True,
            free_bits=1.0,
            use_multiscale_stft=True,
        )
        
        # Inside training loop:
        loss_dict = calculator.compute_all_losses(
            y_hat=y_hat, wave=wave, 
            y_d_hat_r=y_d_hat_r, y_d_hat_g=y_d_hat_g,
            fmap_r=fmap_g, fmap_g=fmap_g,
            z_p=z_p, logs_q=logs_q, m_p=m_p, 
            logs_p=logs_p, z_mask=z_mask
        )
        total_loss = calculator.weighted_total(loss_dict)
    """
    
    def __init__(
        self,
        config=None,
        # Loss weights (typically from config)
        lambda_adv: float = 1.0,
        lambda_fm: float = 1.0,
        lambda_mel: float = 1.0,
        lambda_kl: float = 1.0,
        lambda_energy: float = 0.0,
        # New loss toggles
        use_phase_loss: bool = False,
        lambda_phase: float = 1.0,
        use_envelope_loss: bool = False,
        lambda_envelope: float = 1.0,
        use_free_bits_kl: bool = False,
        free_bits: float = 0.0,
        use_multiscale_stft: bool = False,
        lambda_stft: float = 1.0,
        # STFT params for phase/multi-scale loss
        stft_fft_size: int = 1024,
        stft_hop_size: int = 256,
        stft_win_size: int = 1024,
        # Discriminator settings
        disc_version: str = "v2",
        disc_scale: float = 0.25,
    ):
        # Store weights
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.lambda_kl = lambda_kl
        self.lambda_energy = lambda_energy
        self.lambda_phase = lambda_phase
        self.lambda_envelope = lambda_envelope
        self.lambda_stft = lambda_stft
        
        # Store toggles
        self.use_phase_loss = use_phase_loss
        self.use_envelope_loss = use_envelope_loss
        self.use_free_bits_kl = use_free_bits_kl
        self.free_bits = free_bits
        self.use_multiscale_stft = use_multiscale_stft
        self.disc_version = disc_version
        self.disc_scale = disc_scale
        
        # STFT parameters
        self.stft_fft_size = stft_fft_size
        self.stft_hop_size = stft_hop_size
        self.stft_win_size = stft_win_size
        
        # Initialize multi-scale STFT loss if needed
        self._stft_loss = None
        if use_multiscale_stft:
            self._stft_loss = MultiScaleSTFTLoss()
    
    def compute_discriminator_loss(self, disc_real, disc_fake):
        """Compute discriminator loss (scaled or standard based on version)."""
        if self.disc_version == "v3":
            return discriminator_loss_scaled(disc_real, disc_fake, scale=self.disc_scale)
        else:
            return discriminator_loss(disc_real, disc_fake)
    
    def compute_generator_disc_loss(self, disc_outputs):
        """Compute generator adversarial loss (scaled or standard)."""
        if self.disc_version == "v3":
            return generator_loss_scaled(disc_outputs, scale=self.disc_scale)
        else:
            return generator_loss(disc_outputs)
    
    def compute_kl_loss(self, z_p, logs_q, m_p, logs_p, z_mask, z_p2=None):
        """Compute KL loss (standard or free-bits variant)."""
        if self.use_free_bits_kl:
            return kl_loss_fb(z_p, logs_q, m_p, logs_p, z_mask, z_p2=z_p2, free_bits=self.free_bits)
        else:
            return kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
    
    def compute_phase_loss(self, wave, y_hat):
        """
        Compute phase loss for ISTFT-based vocoders.
        
        Args:
            wave: Target waveform [B, 1, T]
            y_hat: Generated waveform [B, 1, T]
            
        Returns:
            Phase loss tensor (scalar), or 0 if disabled
        """
        if not self.use_phase_loss:
            return torch.tensor(0.0, device=wave.device)
        
        # Compute STFTs
        window = torch.hann_window(self.stft_win_size, device=wave.device, dtype=wave.dtype)
        x_fft = torch.stft(
            wave.squeeze(1) if wave.dim() == 3 else wave,
            self.stft_fft_size, self.stft_hop_size, self.stft_win_size, window,
            return_complex=True, center=False
        )
        g_fft = torch.stft(
            y_hat.squeeze(1) if y_hat.dim() == 3 else y_hat,
            self.stft_fft_size, self.stft_hop_size, self.stft_win_size, window,
            return_complex=True, center=False
        )
        
        return phase_loss(x_fft, g_fft) * self.lambda_phase
    
    def compute_envelope_loss(self, wave, y_hat):
        """
        Compute envelope loss for amplitude matching.
        
        Args:
            wave: Target waveform [B, 1, T]
            y_hat: Generated waveform [B, 1, T]
            
        Returns:
            Envelope loss tensor (scalar), or 0 if disabled
        """
        if not self.use_envelope_loss:
            return torch.tensor(0.0, device=wave.device)
        
        return envelope_loss(wave, y_hat) * self.lambda_envelope
    
    def compute_multiscale_stft_loss(self, wave, y_hat):
        """
        Compute multi-scale STFT loss.
        
        Args:
            wave: Target waveform [B, T] or [B, 1, T]
            y_hat: Generated waveform [B, T] or [B, 1, T]
            
        Returns:
            STFT loss tensor (scalar), or 0 if disabled
        """
        if not self.use_multiscale_stft or self._stft_loss is None:
            return torch.tensor(0.0, device=wave.device)
        
        return self._stft_loss(wave, y_hat) * self.lambda_stft
