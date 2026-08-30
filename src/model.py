"""Model definitions for the CIFAR-10 image classifier.

Supports two architectures:
  - "resnet18": a torchvision ResNet-18, adapted for 32x32 CIFAR-style
    inputs (the default 7x7/stride-2 stem + maxpool is replaced with a
    3x3/stride-1 stem, which is the standard trick for using ImageNet
    architectures on small images).
  - "simple_cnn": a small hand-rolled CNN, useful for fast local testing
    on CPU/no-GPU environments.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class SimpleCNN(nn.Module):
    """A compact CNN for 32x32x3 images (CIFAR-10 / Fashion-MNIST-sized)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _build_cifar_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    """Build a ResNet-18 adapted for small (32x32) inputs."""
    weights = "IMAGENET1K_V1" if pretrained else None
    model = resnet18(weights=weights)

    # Adapt the stem for small images: the default 7x7 stride-2 conv +
    # 3x3 stride-2 maxpool throws away too much spatial resolution for
    # 32x32 inputs. Replace with a 3x3 stride-1 conv and drop the maxpool.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
) -> nn.Module:
    """Factory used by train.py / serve.py.

    Args:
        architecture: "resnet18" or "simple_cnn".
        num_classes: number of output classes.
        pretrained: if True and architecture == "resnet18", start from
            ImageNet weights (fine-tuning). Kept False by default so
            training works fully offline in CI/containers.
    """
    architecture = architecture.lower()
    if architecture == "resnet18":
        return _build_cifar_resnet18(num_classes=num_classes, pretrained=pretrained)
    if architecture in ("simple_cnn", "cnn"):
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {architecture!r}")
