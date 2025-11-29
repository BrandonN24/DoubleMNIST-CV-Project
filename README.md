# Project Dependencies
This project relies on the following packages
- argparse
- pytorch
- torchvision
- numpy
- tqdm
- matplotlib
- imageio
- os
- urllib
- gzip
- shutil

# Using the dataset generator
Run the following command to download the MNIST dataset and generate the DoubleMNIST dataset
`python generator.py --num_image_per_class 1000 --multimnist_path ./dataset/double_mnist --num_digit 2 --image_size 64 64`