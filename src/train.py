import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

from dataset import get_dataloaders
from model import get_model
from utils import train_one_epoch, evaluate


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader, num_classes = get_dataloaders()

    model = get_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=5e-4
    )

    epochs = 15

    # Cosine annealing smoothly decays LR to near-zero
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision scaler (only effective on CUDA)
    scaler = GradScaler() if device.type == "cuda" else None

    best_val_acc = 0

    for epoch in range(epochs):

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} (LR: {current_lr:.6f})")

        train_loss, train_top1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        val_loss, val_top1 = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_top1:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_top1:.2f}%")

        # Save ONLY best model
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Saved Best Model! (Val Acc: {val_top1:.2f}%)")

    print(f"\nTraining completed. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()