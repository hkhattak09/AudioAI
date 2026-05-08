"""Training loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def tacotron2_loss(
    mel_pred: torch.Tensor,
    mel_pred_postnet: torch.Tensor,
    stop_pred: torch.Tensor,
    mel_target: torch.Tensor,
    stop_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tacotron2 training losses.

    Args:
        mel_pred: [B, n_mels, T]
        mel_pred_postnet: [B, n_mels, T]
        stop_pred: [B, T]
        mel_target: [B, n_mels, T]
        stop_target: [B, T]

    Returns:
        mel_loss, mel_postnet_loss, stop_loss
    """
    mel_loss = F.mse_loss(mel_pred, mel_target)
    mel_postnet_loss = F.mse_loss(mel_pred_postnet, mel_target)
    stop_loss = F.binary_cross_entropy_with_logits(stop_pred, stop_target)
    return mel_loss, mel_postnet_loss, stop_loss


def feature_matching_loss(fmap_r: list[list[torch.Tensor]], fmap_g: list[list[torch.Tensor]]) -> torch.Tensor:
    """HiFiGAN feature matching loss.

    Args:
        fmap_r: list of discriminator feature maps for real audio
        fmap_g: list of discriminator feature maps for generated audio
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += F.l1_loss(gl, rl.detach())
    return loss


def discriminator_hinge_loss(real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for discriminator."""
    loss = 0
    for dr, dg in zip(real_logits, fake_logits):
        loss += torch.mean(F.relu(1 - dr)) + torch.mean(F.relu(1 + dg))
    return loss


def generator_adversarial_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Generator adversarial loss."""
    loss = 0
    for dg in fake_logits:
        loss += -torch.mean(dg)
    return loss


def mel_reconstruction_loss_placeholder(mel_pred: torch.Tensor, mel_target: torch.Tensor) -> torch.Tensor:
    """Placeholder mel reconstruction loss for vocoder training."""
    return F.l1_loss(mel_pred, mel_target)
