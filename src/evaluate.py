# evaluate_saved_models.py

import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import get_model
from utils import evaluate


def evaluate_model(model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader, num_classes = get_dataloaders()

    # Load model
    model = get_model(num_classes)
    model = model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    loss, top1 = evaluate(model, test_loader, criterion, device)

    print(f"\nResults for {model_path}")
    print(f"Loss: {loss:.4f}")
    print(f"Top-1 Accuracy: {top1:.2f}%")
    print("-" * 50)


if __name__ == "__main__":

    evaluate_model("best_model.pth")
