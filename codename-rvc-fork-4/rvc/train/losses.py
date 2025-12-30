from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

def phase_loss(x_fft: torch.Tensor, g_fft: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
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


def feature_loss(fmap_r, fmap_g):
    """
    Compute the feature loss between reference and generated feature maps.

    Args:
        fmap_r (list of torch.Tensor): List of reference feature maps.
        fmap_g (list of torch.Tensor): List of generated feature maps.
    """
    return 2 * sum(
        torch.mean(torch.abs(rl - gl))
        for dr, dg in zip(fmap_r, fmap_g)
        for rl, gl in zip(dr, dg)
    )


def feature_loss_mask(fmap_r, fmap_g, silence_mask=None, reduce=True):
    """
    Silence-aware feature matching loss.
    If silence_mask is provided, applies it per sample to reduce loss contribution.

    Args:
        fmap_r (List[List[Tensor]]): Feature maps from real audio
        fmap_g (List[List[Tensor]]): Feature maps from generated audio
        silence_mask (Tensor or None): Tensor of shape [B], 1 for voiced, 0 for silence
        reduce (bool): Whether to return mean or per-sample loss
    Returns:
        Scalar loss or per-sample tensor
    """
    losses = []

    for dr, dg in zip(fmap_r, fmap_g):  # across discriminators
        for rl, gl in zip(dr, dg):      # across layers
            diff = torch.abs(rl - gl)
            per_sample = diff.view(diff.shape[0], -1).mean(dim=1)  # [B]
            losses.append(per_sample)

    total = torch.stack(losses, dim=0).mean(dim=0)  # mean over layers → [B]

    if silence_mask is not None:
        total = total * silence_mask  # scale loss per sample

    if reduce:
        return total.sum() / (silence_mask.sum() + 1e-6 if silence_mask is not None else total.numel())
    else:
        return total  # shape [B]


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Compute the discriminator loss for real and generated outputs.

    Args:
        disc_real_outputs (list of torch.Tensor): List of discriminator outputs for real samples.
        disc_generated_outputs (list of torch.Tensor): List of discriminator outputs for generated samples.
    """
    loss = 0
    # r_losses = []
    # g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)

        # r_losses.append(r_loss.item())
        # g_losses.append(g_loss.item())
        loss += r_loss + g_loss

    return loss # , r_losses, g_losses


def generator_loss(disc_outputs):
    """
    LSGAN Generator Loss:
    """
    loss = 0
    #gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        # gen_losses.append(l.item())
        loss += l

    return loss #, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Compute the Kullback-Leibler divergence loss.

    Args:
        z_p (torch.Tensor): Sampled latent variable transformed by the flow [b, h, t_t].
        logs_q (torch.Tensor): Log variance of the posterior distribution q [b, h, t_t].
        m_p (torch.Tensor): Mean of the prior distribution p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of the prior distribution p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()

    return loss

def kl_loss_clamped(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Compute the Kullback-Leibler divergence loss.
    Variant with non-negativity clamp.

    Args:
        z_p (torch.Tensor): Sampled latent variable transformed by the flow [b, h, t_t].
        logs_q (torch.Tensor): Log variance of the posterior distribution q [b, h, t_t].
        m_p (torch.Tensor): Mean of the prior distribution p [b, h, t_t].
        logs_p (torch.Tensor): Log variance of the prior distribution p [b, h, t_t].
        z_mask (torch.Tensor): Mask for the latent variables [b, h, t_t].
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = (kl * z_mask).sum()
    loss = kl / z_mask.sum()
    loss = torch.clamp(loss, min=0.0)

    return loss


def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    tau = 0.04
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()

        # Median centering
        m_DG = torch.median((dr - dg))

        # Relative difference
        # We only penalize when the Real sample is NOT sufficiently better than Fake
        # Condition: Real < Fake + Margin
        diff = (dr - dg) - m_DG
        mask = dr < (dg + m_DG)

        # Calculate Squared Error on the masked (hard) examples
        # We use empty-safe mean calculation
        masked_diff = diff[mask]
        if masked_diff.numel() > 0:
            L_rel = torch.mean(masked_diff ** 2)
        else:
            L_rel = torch.tensor(0.0, device=dr.device)

        # Truncate the loss (clamp)
        loss += tau - F.relu(tau - L_rel)

    return loss


def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    tau = 0.04
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        
        # Median centering (Fake - Real)
        diff = dg - dr 
        m_DG = torch.median(diff)

        # Relative difference
        # Generator wants Fake > Real. 
        # We penalize when Fake is NOT sufficiently better than Real
        # Condition: Fake < Real + Margin
        rel_diff = diff - m_DG
        mask = diff < m_DG

        masked_diff = rel_diff[mask]
        if masked_diff.numel() > 0:
            L_rel = torch.mean(masked_diff ** 2)
        else:
            L_rel = torch.tensor(0.0, device=dg.device)

        # Truncate the loss (clamp)
        loss += tau - F.relu(tau - L_rel)

    return loss

def discriminator_loss_v2(disc_real_outputs, disc_generated_outputs):
    """
    Compute the discriminator loss for real and generated outputs.
    """
    loss = 0

    # LSGAN Loss
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr.float()) ** 2)
        g_loss = torch.mean(dg.float() ** 2)
        loss += r_loss + g_loss

    # TPRLS Loss
    loss_rel = discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs)
    loss += loss_rel

    return loss


def generator_loss_v2(disc_outputs, disc_real_outputs):
    """
    LSGAN Generator Loss + TPRLS
    """
    loss = 0

    # Existing LSGAN Loss
    for dg in disc_outputs:
        l = torch.mean((1 - dg.float()) ** 2)
        loss += l

    # TPRLS Loss
    loss_rel = generator_TPRLS_loss(disc_real_outputs, disc_outputs)
    loss += loss_rel

    return loss

class HingeAdversarialLoss(nn.Module):
    """Module for calculating adversarial loss in GANs."""

    def __init__(self,) -> None:
        """
        Hinge adversarial loss.
        """
        super().__init__()

        self.adv_criterion = self._hinge_adv_loss
        self.fake_criterion = self._hinge_fake_loss
        self.real_criterion = self._hinge_real_loss

    def forward(
        self, p_fakes: List[Tensor], p_reals: Optional[List[Tensor]] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Calculate adversarial loss for both generator and discriminator.

        Args:
            p_fakes (List[Tensor]): List of discriminator outputs from the generated data.
            p_reals (List[Tensor], optional): List of discriminator outputs from real data.
                If not provided, only generator loss is computed (default: None).

        Returns:
            Tensor: Generator adversarial loss.
            If p_reals is provided:
                Tuple[Tensor, Tensor]: Fake and real discriminator loss values.
        """
        # Generator adversarial loss
        if p_reals is None:
            adv_loss = 0.0
            for p_fake in p_fakes:
                adv_loss += self.adv_criterion(p_fake)

            return adv_loss

        # Discriminator adversarial loss
        else:
            fake_loss, real_loss = 0.0, 0.0
            for p_fake, p_real in zip(p_fakes, p_reals):
                fake_loss += self.fake_criterion(p_fake)
                real_loss += self.real_criterion(p_real)

            return fake_loss, real_loss


    def _hinge_adv_loss(self, x: Tensor) -> Tensor:
        """Calculate hinge loss for generator."""
        return -x.mean()

    def _hinge_real_loss(self, x: Tensor) -> Tensor:
        """Calculate hinge loss for real samples."""
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x: Tensor) -> Tensor:
        """Calculate hinge loss for fake samples."""
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))


def envelope_loss(y_real, y_fake, 
                  pool=nn.MaxPool1d(kernel_size=5, stride=3), 
                  criterion=nn.L1Loss()):
    """
    Calculates the envelope loss between real and generated audio.
    Matches volume peaks and troughs to improve transient clarity.
    """

    # Calculate loss for both polarities (peaks and troughs)
    loss_pos = criterion(pool(y_real), pool(y_fake))
    loss_neg = criterion(pool(-y_real), pool(-y_fake))
    
    return loss_pos + loss_neg
