import torch
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()

    running_loss = 0.0
    correct_top1 = 0
    total = 0

    use_amp = scaler is not None and device.type == "cuda"

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        correct_top1 += predicted.eq(labels).sum().item()
        total += labels.size(0)

    top1 = 100 * correct_top1 / total
    return running_loss / len(loader), top1


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct_top1 = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(labels).sum().item()
            total += labels.size(0)

    top1 = 100 * correct_top1 / total
    return running_loss / len(loader), top1