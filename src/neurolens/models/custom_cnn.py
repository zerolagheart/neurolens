"""A compact CNN trained from scratch, used as a transparent baseline next to transfer learning."""

from __future__ import annotations

import torch
from torch import nn


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> list[nn.Module]:
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]


class ConvBlock(nn.Sequential):
    """Two 3x3 conv-BN-ReLU layers, optionally followed by 2x2 max pooling."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool) -> None:
        layers = _conv_bn_relu(in_ch, out_ch) + _conv_bn_relu(out_ch, out_ch)
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class NeuroCNN(nn.Module):
    """Stem (stride 2) + four conv blocks + global average pooling + linear classifier.

    For a 224x224 input the last block produces 14x14 feature maps, which is also the Grad-CAM
    resolution (``features[-1]``). About 1.2M parameters.
    """

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        stem_ch = channels[0]
        blocks: list[nn.Module] = [nn.Sequential(*_conv_bn_relu(3, stem_ch, stride=2))]
        in_ch = stem_ch
        for idx, out_ch in enumerate(channels):
            blocks.append(ConvBlock(in_ch, out_ch, pool=idx < len(channels) - 1))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))
