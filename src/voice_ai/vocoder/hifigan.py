"""HiFiGAN vocoder modules.

Reference architectures:
- https://github.com/jik876/hifi-gan
- https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20Audio/HIFIGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class HiFiGANConfig:
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates: tuple[int, ...] = (8, 8, 2, 2)
    upsample_initial_channel: int = 128
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4)
    model_in_channels: int = 80
    model_out_channels: int = 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=self._get_padding(kernel_size, d))
            )
            self.convs2.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=self._get_padding(kernel_size, 1))
            )

    def _get_padding(self, kernel_size: int, dilation: int) -> int:
        return (kernel_size * dilation - dilation) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    def __init__(self, config: HiFiGANConfig):
        super().__init__()
        self.config = config
        self.conv_pre = nn.Conv1d(config.model_in_channels, config.upsample_initial_channel, 7, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2 ** i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(ResidualBlock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, config.model_out_channels, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_mels, T]
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = 0
            for j in range(len(self.config.resblock_kernel_sizes)):
                xs = xs + self.resblocks[i * len(self.config.resblock_kernel_sizes) + j](x)
            x = xs / len(self.config.resblock_kernel_sizes)
        x = F.leaky_relu(x)
        x = torch.tanh(self.conv_post(x))
        return x  # [B, 1, T']


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # x: [B, 1, T]
        fmap = []
        # Pad to multiple of period
        if x.shape[-1] % self.period != 0:
            n_pad = self.period - (x.shape[-1] % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
        x = x.view(x.shape[0], x.shape[1], x.shape[-1] // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> tuple:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, padding=20, groups=4)),
                norm_f(nn.Conv1d(256, 512, 41, 4, padding=20, groups=16)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, padding=20, groups=64)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, padding=20, groups=256)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> tuple:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class HiFiGAN(nn.Module):
    def __init__(self, config: HiFiGANConfig | None = None):
        super().__init__()
        self.config = config or HiFiGANConfig()
        self.generator = Generator(self.config)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def generate(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Args:
            mel: [B, n_mels, T]

        Returns:
            waveform: [B, 1, T']
        """
        return self.generator(mel)

    def discriminate_for_discriminator(
        self, real_audio: torch.Tensor, fake_audio: torch.Tensor
    ) -> dict:
        """Run discriminators with detached fake audio (for D training).

        Args:
            real_audio: [B, 1, T']
            fake_audio: [B, 1, T']

        Returns:
            Dict with mpd and msd discriminator outputs.
        """
        fake_audio_detached = fake_audio.detach()
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.mpd(real_audio, fake_audio_detached)
        y_d_rs2, y_d_gs2, fmap_rs2, fmap_gs2 = self.msd(real_audio, fake_audio_detached)
        return {
            "mpd": (y_d_rs, y_d_gs, fmap_rs, fmap_gs),
            "msd": (y_d_rs2, y_d_gs2, fmap_rs2, fmap_gs2),
        }

    def discriminate_for_generator(
        self, real_audio: torch.Tensor, fake_audio: torch.Tensor
    ) -> dict:
        """Run discriminators with non-detached fake audio (for G training).

        Args:
            real_audio: [B, 1, T']
            fake_audio: [B, 1, T']

        Returns:
            Dict with mpd and msd discriminator outputs.
        """
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.mpd(real_audio, fake_audio)
        y_d_rs2, y_d_gs2, fmap_rs2, fmap_gs2 = self.msd(real_audio, fake_audio)
        return {
            "mpd": (y_d_rs, y_d_gs, fmap_rs, fmap_gs),
            "msd": (y_d_rs2, y_d_gs2, fmap_rs2, fmap_gs2),
        }
