import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_transfer_resnet18(num_classes=10, pretrained=True, frozen=False):
    """
    torchvision ResNet-18 used for both transfer-learning variants.

    frozen=False -> full fine-tuning
    frozen=True  -> frozen backbone + trainable 512->10 head
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if frozen:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
