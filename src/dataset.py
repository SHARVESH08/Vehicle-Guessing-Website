from torchvision import datasets
from torch.utils.data import DataLoader
from transforms import train_transforms, test_transforms


def get_dataloaders(data_dir="../Dataset", batch_size=32):

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transforms
    )

    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=test_transforms
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    num_classes = len(train_dataset.classes)

    return train_loader, test_loader, num_classes