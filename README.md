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

# Running the main driver script
To run the main driver script, run the following command:
`python trainNet.py`

There are several arguments that can be used to modify which model architecture is selected and adjust some hyperparameters.
`--mode` selects the model architecture to be used for training and testing.
- 1 - AlexNet-like Architecture (default)
- 2 - ResNet18-like Architecture
- 3 - Fully Convolutional Architecture

`--learning_rate` sets the learning rate for the model.
- default is 0.0005

`--num_epochs` sets the number of epochs to train the model for
- default is 30

`--batch_size` sets the batch-size for each iteration during training and testing.
- default is 50
- Values must be divide evenly into dataset sizes.



## Internal dependencies
This project has internal dependencies that are located in the 'packages' folder.
These packages include the dataloader.py, networks.py, and plot.py files. If these modules are removed from the 'packages' folder, the main driver script will not work.