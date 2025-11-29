import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data(img_dir):
    train_dir = img_dir + "/train"
    test_dir = img_dir + "/test"
    val_dir = img_dir + "/val"

    mean, std = 0.1307, 0.3081  # MNIST defaults


    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=15),                                                  # small rotations
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),   # random shifts
        transforms.ToTensor(),                                                                  # convert to tensor
        transforms.Normalize((mean,), (std,)),                                                   # normalize
        transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))], p=0.5)     # erase patches
    ])

    # using 64 x 64 images already, so just convert to tensor
    data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    # Using image folder to create datasets

    # Grab images from directory and apply transform.
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transform,
                                      target_transform=None)
    
    # Grab images from direcotry and apply transform.
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform)
    
    val_data = datasets.ImageFolder(root=val_dir,
                                    transform=data_transform)
    
    return train_data, test_data, val_data

def get_double_MNIST(img_dir, FLAGS):
    # Load data from the dataset folder
    train_data, test_data, val_data = load_data(img_dir)
    
    # Turn train, validation, and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=FLAGS.batch_size, # how many samples per batch
                                num_workers=4, # how many subprocesses to use for data loading
                                shuffle=True) # shuffle the data between epochs

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=FLAGS.batch_size, 
                                num_workers=4, 
                                shuffle=False) # don't usually need to shuffle testing data
    
    val_dataloader= DataLoader(dataset=val_data,
                               batch_size=FLAGS.batch_size,
                               num_workers=4,
                               shuffle=False) # don't need to shuffle validation data either.

    return train_dataloader, test_dataloader, val_dataloader
