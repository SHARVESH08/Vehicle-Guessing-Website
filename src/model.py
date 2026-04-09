import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def get_model(num_classes):

    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Only unfreeze the last block (block 7) — less fine-tuning = less overfitting
    for param in model.features[7].parameters():
        param.requires_grad = True

    # Replace classifier head with stronger dropout
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model