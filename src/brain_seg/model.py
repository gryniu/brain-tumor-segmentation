"""3D U-Net for volumetric multi-class segmentation.

``UNet3D`` follows the original 3D U-Net (Cicek et al., 2016) with modern
defaults (instance normalisation, LeakyReLU), written out explicitly so every
stage is easy to read and modify::

    input (B, C_in, X, Y, Z)
      |
      enc0: ConvBlock(C_in -> f0)  ------------------------------ skip0 ---+
      maxpool /2                                                           |
      enc1: ConvBlock(f0 -> f1)    -------------------- skip1 ---+         |
      maxpool /2                                                 |         |
      ...                                                        |         |
      bottleneck: ConvBlock(f[n-2] -> f[n-1])  (lowest resolution)         |
      ...                                                        |         |
      up1: ConvTranspose x2 (f2 -> f1), concat skip1, ConvBlock(2*f1 -> f1)|
      up0: ConvTranspose x2 (f1 -> f0), concat skip0, ConvBlock(2*f0 -> f0)
      head: 1x1x1 Conv (f0 -> K)
      |
    logits (B, K, X, Y, Z)   -- softmax over K classes, argmax = label map

With ``features = [16, 32, 64, 128, 256]`` there are 4 downsampling steps, so
every spatial input dimension must be divisible by 2**4 = 16.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


def _norm(norm: str, channels: int) -> nn.Module:
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    if norm == "group":
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"Unknown normalisation {norm!r}")


class ConvBlock(nn.Sequential):
    """Two (3x3x3 conv -> norm -> LeakyReLU) layers; keeps spatial size."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance", dropout: float = 0.0):
        layers: list[nn.Module] = []
        for i in range(2):
            layers += [
                nn.Conv3d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1, bias=False),
                _norm(norm, out_channels),
                nn.LeakyReLU(0.01, inplace=True),
            ]
            if dropout > 0 and i == 0:
                layers.append(nn.Dropout3d(dropout))
        super().__init__(*layers)


class UNet3D(nn.Module):
    """Encoder-decoder 3D U-Net with skip connections.

    Args:
        in_channels: number of input MRI modalities (4 for T1/T1ce/T2/FLAIR).
        out_channels: number of output classes (4: BG, NCR, ED, ET).
        features: channels per resolution level; the last entry is the bottleneck.
        norm: ``instance`` (default, robust for small batches), ``batch`` or ``group``.
        dropout: dropout probability inside each conv block (0 disables).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: tuple[int, ...] | list[int] = (16, 32, 64, 128, 256),
        norm: str = "instance",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        features = list(features)
        if len(features) < 2:
            raise ValueError("UNet3D needs at least two feature levels")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_downsamplings = len(features) - 1

        self.encoders = nn.ModuleList()
        channels = in_channels
        for f in features[:-1]:
            self.encoders.append(ConvBlock(channels, f, norm, dropout))
            channels = f
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(features[-2], features[-1], norm, dropout)

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f_in, f_skip in zip(reversed(features[1:]), reversed(features[:-1])):
            self.upsamplers.append(nn.ConvTranspose3d(f_in, f_skip, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(2 * f_skip, f_skip, norm, dropout))
        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input of shape (B, {self.in_channels}, X, Y, Z), got {tuple(x.shape)}"
            )
        factor = 2**self.num_downsamplings
        if any(s % factor for s in x.shape[2:]):
            raise ValueError(f"Spatial size {tuple(x.shape[2:])} must be divisible by {factor}")

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upsample, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            x = upsample(x)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.head(x)


def build_model(cfg: ModelConfig, in_channels: int, out_channels: int) -> nn.Module:
    """Create the configured network. ``monai_unet`` is MONAI's residual U-Net, for comparison."""
    if cfg.name == "unet3d":
        return UNet3D(in_channels, out_channels, cfg.features, cfg.norm, cfg.dropout)
    if cfg.name == "monai_unet":
        from monai.networks.nets import UNet

        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(cfg.features),
            strides=(2,) * (len(cfg.features) - 1),
            num_res_units=2,
            norm=cfg.norm,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown model {cfg.name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
