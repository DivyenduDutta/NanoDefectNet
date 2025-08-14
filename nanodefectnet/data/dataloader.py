import os
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nanodefectnet.utils.constants import ACTUAL_FAILURE_TYPE_TO_ID
from nanodefectnet.utils.logger import LoggerConfig

LOGGER = LoggerConfig().logger


def synchronize_labels(dataset: datasets.ImageFolder):
    """
    Synchronize the labels as per `ACTUAL_FAILURE_TYPE_TO_ID` mapping.
    This is required because the labels as per `ImageFolder` follow folder alphabetical order
    """
    dataset.targets = [
        ACTUAL_FAILURE_TYPE_TO_ID[dataset.classes[label]] for label in dataset.targets
    ]
    dataset.class_to_idx = ACTUAL_FAILURE_TYPE_TO_ID
    dataset.classes = list(ACTUAL_FAILURE_TYPE_TO_ID.keys())


def create_loaders(cfg):
    dataset_path = cfg["dataset"]["root_dataset_path"]
    batch_size = cfg["dataset"]["batch_size"]

    do_resize = cfg["data"].get("do_resize", True)
    if do_resize:
        basic_transform = transforms.Compose(
            [
                transforms.Resize((cfg["data"]["size"], cfg["data"]["size"])),
                transforms.ToTensor(),
            ]
        )
    else:
        basic_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    # 1 - Instantiate the dataset for train, valid and test
    wafermap_dataset_train = datasets.ImageFolder(
        os.path.join(os.getcwd(), dataset_path, cfg["dataset"]["path_dataset_train"]),
        transform=basic_transform,
    )
    synchronize_labels(wafermap_dataset_train)
    wafermap_dataset_val = datasets.ImageFolder(
        os.path.join(os.getcwd(), dataset_path, cfg["dataset"]["path_dataset_val"]),
        transform=basic_transform,
    )
    synchronize_labels(wafermap_dataset_val)
    wafermap_dataset_test = datasets.ImageFolder(
        os.path.join(os.getcwd(), dataset_path, cfg["dataset"]["path_dataset_test"]),
        transform=basic_transform,
    )
    synchronize_labels(wafermap_dataset_test)

    # 2 - Instantiate the dataloaders
    wafermap_dataloader_train = DataLoader(
        dataset=wafermap_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )

    wafermap_dataloader_val = DataLoader(
        dataset=wafermap_dataset_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )

    wafermap_dataloader_test = DataLoader(
        dataset=wafermap_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    return wafermap_dataloader_train, wafermap_dataloader_val, wafermap_dataloader_test


# Function to count labels in a DataLoader
def _count_labels_from_loader(loader):
    label_counts = Counter()
    for _, labels in loader:
        # If labels is a tensor, move to CPU and flatten
        labels = labels.view(-1).cpu().numpy()
        label_counts.update(labels)
    return label_counts


def get_class_weights(cfg):
    train_dataloader, val_dataloader, test_dataloader = create_loaders(cfg)
    # Count labels from all loaders ie, full dataset
    total_counts = Counter()
    for loader in [train_dataloader, val_dataloader, test_dataloader]:
        total_counts.update(_count_labels_from_loader(loader))

    num_classes = len(total_counts)
    total_samples = sum(total_counts.values())

    # Calculate inverse frequency weights
    class_weights = []
    for i in range(num_classes):
        weight = total_samples / (num_classes * total_counts.get(i, 1))
        class_weights.append(weight)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    weights_str = ", ".join(f"{w:.4f}" for w in class_weights_tensor)
    LOGGER.debug(f"Class weights: {weights_str}")
    if torch.cuda.is_available():
        class_weights_tensor = class_weights_tensor.cuda()

    return class_weights_tensor
