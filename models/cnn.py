"""
Model 1 — Simple CNN
=====================
A custom convolutional network trained from scratch on EuroSAT.

Architecture (matches the trained checkpoint's actual module structure —
two named nn.Sequential blocks, ``features`` and ``classifier``):

    features (Sequential, indices 0-15):
      0  Conv2D(3->32, 3x3, pad=1)
      1  BatchNorm2D(32)
      2  ReLU
      3  MaxPool2D(2x2)
      4  Conv2D(32->64, 3x3, pad=1)
      5  BatchNorm2D(64)
      6  ReLU
      7  MaxPool2D(2x2)
      8  Conv2D(64->128, 3x3, pad=1)
      9  BatchNorm2D(128)
      10 ReLU
      11 MaxPool2D(2x2)
      12 Conv2D(128->256, 3x3, pad=1)
      13 BatchNorm2D(256)
      14 ReLU
      15 MaxPool2D(2x2)

    classifier (Sequential, indices 0-5):
      0  AdaptiveAvgPool2D(1x1)
      1  Flatten
      2  Linear(256 -> 128)
      3  ReLU
      4  Dropout(0.5)
      5  Linear(128 -> 10)

No residual connections. Four convolutional stages with BatchNorm and
adaptive global average pooling before the classifier head.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Custom CNN trained from scratch for EuroSAT 10-class classification."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),   # 0
            nn.BatchNorm2d(32),                                     # 1
            nn.ReLU(inplace=True),                                  # 2
            nn.MaxPool2d(2, 2),                                     # 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),            # 4
            nn.BatchNorm2d(64),                                     # 5
            nn.ReLU(inplace=True),                                  # 6
            nn.MaxPool2d(2, 2),                                     # 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # 8
            nn.BatchNorm2d(128),                                    # 9
            nn.ReLU(inplace=True),                                  # 10
            nn.MaxPool2d(2, 2),                                     # 11
            nn.Conv2d(128, 256, kernel_size=3, padding=1),          # 12
            nn.BatchNorm2d(256),                                    # 13
            nn.ReLU(inplace=True),                                  # 14
            nn.MaxPool2d(2, 2),                                     # 15
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),      # 0
            nn.Flatten(),                      # 1
            nn.Linear(256, 128),               # 2
            nn.ReLU(inplace=True),             # 3
            nn.Dropout(0.5),                   # 4
            nn.Linear(128, num_classes),       # 5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int = 10) -> SimpleCNN:
    """Factory used by the model loader to instantiate this architecture."""
    return SimpleCNN(num_classes=num_classes)
