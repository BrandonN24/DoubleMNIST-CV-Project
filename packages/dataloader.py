import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def make_dataset(root, transform):
    dataset = datasets.ImageFolder(root=root, transform=transform)
    # Override target_transform: map index → folder name → int
    dataset.target_transform = lambda idx: int(dataset.classes[idx])
    return dataset

def load_data(img_dir):
    train_dir = os.path.join(img_dir, "train")
    test_dir  = os.path.join(img_dir, "test")
    val_dir   = os.path.join(img_dir, "val")

    mean, std = 0.1307, 0.3081  # MNIST defaults

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.9, 1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
        transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))], p=0.5)
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    train_data = make_dataset(train_dir, train_transform)
    val_data   = make_dataset(val_dir, test_transform)
    test_data  = make_dataset(test_dir, test_transform)

    return train_data, test_data, val_data

def get_double_MNIST(img_dir, FLAGS):
    train_data, test_data, val_data = load_data(img_dir)

    train_dataloader = DataLoader(train_data,
                                  batch_size=FLAGS.batch_size,
                                  num_workers=4,
                                  shuffle=True)

    val_dataloader = DataLoader(val_data,
                                batch_size=FLAGS.batch_size,
                                num_workers=4,
                                shuffle=False)

    test_dataloader = DataLoader(test_data,
                                 batch_size=FLAGS.batch_size,
                                 num_workers=4,
                                 shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader