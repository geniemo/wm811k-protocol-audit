"""The one fixed model. No tuning happens here or anywhere else."""
import torch
from torch import nn


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    """conv(32)-pool-conv(64)-pool-conv(128)-pool-conv(128)-GAP-dropout-linear. ~0.24M params."""

    def __init__(self, n_classes: int, in_ch: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_ch, 32), nn.MaxPool2d(2),
            _block(32, 64), nn.MaxPool2d(2),
            _block(64, 128), nn.MaxPool2d(2),
            _block(128, 128), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ResNet18Adapted(nn.Module):
    """torchvision resnet18 from scratch; 3x3 stride-1 stem and no stem max-pool so 64x64 inputs keep an 8x8 final map."""

    def __init__(self, n_classes: int, in_ch: int = 3):
        super().__init__()
        from torchvision.models import resnet18
        net = resnet18(weights=None, num_classes=n_classes)
        net.conv1 = nn.Conv2d(in_ch, 64, 3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(name: str, n_classes: int) -> nn.Module:
    if name == "smallcnn":
        return SmallCNN(n_classes)
    if name == "resnet18":
        return ResNet18Adapted(n_classes)
    raise ValueError(f"unknown model {name!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
