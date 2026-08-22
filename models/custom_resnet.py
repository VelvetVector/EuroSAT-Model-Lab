"""
Model 2 — Custom ResNet-18 (Trained From Scratch)
===================================================
A hand-written ResNet-18 implementation, distinct from
``torchvision.models.resnet18``. Trained from scratch with random
initialization on EuroSAT. Source notebook: gemini_sol2(1).ipynb

Stem
----
Unlike the standard ImageNet stem (7x7 conv, stride 2, + maxpool), this
custom stem is a single 3x3 convolution:

    3 -> 64, kernel 3x3, stride 1, BatchNorm, ReLU

This is appropriate for the smaller 64x64 EuroSAT input.

Body
----
Four residual stages, each made of 2 BasicBlocks (block config [2, 2, 2, 2]):

    Stage 1: 64  channels
    Stage 2: 128 channels
    Stage 3: 256 channels
    Stage 4: 512 channels

Each BasicBlock:

    Conv2D(3x3) -> BatchNorm -> ReLU -> Conv2D(3x3) -> BatchNorm
        + (identity or 1x1-conv projection skip connection)
    -> ReLU

Head
----
AdaptiveAvgPool2D(1x1) -> Flatten -> Linear(512 -> 10)
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock with identity or projection skip connection."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut is needed whenever spatial size or channel
        # count changes between the block's input and output.
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class CustomResNet18(nn.Module):
    """Custom, from-scratch ResNet-18 for EuroSAT 10-class classification."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3, block_config=(2, 2, 2, 2)):
        super().__init__()

        # Custom stem: 3x3 conv, stride 1 (no 7x7 conv + maxpool like ImageNet ResNets).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.in_channels = 64
        self.layer1 = self._make_stage(64, block_config[0], stride=1)
        self.layer2 = self._make_stage(128, block_config[1], stride=2)
        self.layer3 = self._make_stage(256, block_config[2], stride=2)
        self.layer4 = self._make_stage(512, block_config[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def get_residual_stages(self):
        """Return the four residual stages, useful for architecture visualization."""
        return {
            "layer1 (64ch)": self.layer1,
            "layer2 (128ch)": self.layer2,
            "layer3 (256ch)": self.layer3,
            "layer4 (512ch)": self.layer4,
        }


def build_model(num_classes: int = 10) -> CustomResNet18:
    """Factory used by the model loader to instantiate this architecture."""
    return CustomResNet18(num_classes=num_classes)
