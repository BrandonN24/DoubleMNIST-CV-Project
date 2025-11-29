import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from packages.plot import print_plots
from packages.networks import Network
import packages.dataloader as dataloader
from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    ''' Trains the model for an epoch and performs an optmization step.
    In:
        model: the model to train
        device, 'cuda' or 'cpu
        train_loader: dataloader for training samples
        optimizer: optimizer to use for model parameter updates
        criterion: used to compute loss for prediction and target
        epoch: Current epoch to train for.
        batch_size: batch size for iteration to be used.
    Out:
        train_loss: training loss for the epoch performed.
        train_acc: training accuracy for the epoch performed.'''
    
    # set model to training mode before each epoch
    model.train()

    # create empty list to store losses
    losses = []
    correct = 0

    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(tqdm(train_loader, desc=f"Training Progress", leave=False)):
        data, targets = batch_sample

        # push data to correct device
        data = data.to(device)
        targets = targets.to(device)

        # separate into two digits
        digit1 = targets // 10
        digit2 = targets % 10

        # Reset optimizer gradients. Avoids grad accumulation
        optimizer.zero_grad()

        # Do forward pass for current set of data
        digit1_logits, digit2_logits = model(data)

        # Compute loss based on criterion
        loss_1 = criterion(digit1_logits, digit1)
        loss_2 = criterion(digit2_logits, digit2)

        loss = loss_1 + loss_2

        # Computes gradient based on final loss
        loss.backward()

        # Store loss
        losses.append(loss.item())

        # Optimize model parameters based on learning rate and gradient
        optimizer.step()

        # make digit predictions
        digit1_pred = digit1_logits.argmax(dim=1, keepdim=True)
        digit2_pred = digit2_logits.argmax(dim=1, keepdim=True)

        # Count correct predictions overall
        # sum the instances where our prediction matches the target
        for pred1, pred2, target1, target2 in zip(digit1_pred, digit2_pred, digit1, digit2):
            if pred1 == target1:
                correct += 0.5
            if pred2 == target2:
                correct += 0.5

    train_loss = float(np.mean(losses))
    train_acc = 100. * correct / ((batch_idx+1) * batch_size)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * batch_size,
        100 * correct / ((batch_idx+1) * batch_size)))

    return train_loss, train_acc

def test(model, device, criterion, test_loader):
    ''' Tests the model for an epoch and performs an optmization step.
    In:
        model: the model to train
        device, 'cuda' or 'cpu
        test_loader: dataloader for test/validation samples
        optimizer: optimizer to use for model parameter updates
        criterion: used to compute loss for prediction and target
    Out:
        test_loss: test loss for the current state of the model
        test_acc: test accuracy for the current state of the model
    '''
    # Set model to evaluation mode to notify all layers not to update their parameters.

    losses = []
    correct = 0

    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(test_loader,  desc=f"Testing Progress", leave=False)):
            data, targets = sample

            # push data to correct device
            data = data.to(device)
            targets = targets.to(device)

            # separate into two digits
            digit1 = targets // 10
            digit2 = targets % 10

            # Predict using data by doing forward pass
            digit1_logits, digit2_logits = model(data)

            # compute loss based on same criterion as training
            loss1 = criterion(digit1_logits, digit1)
            loss2 = criterion(digit2_logits, digit2)

            loss = loss1 + loss2

            losses.append(loss.item())

            # make digit predictions
            digit1_pred = digit1_logits.argmax(dim=1, keepdim=True)
            digit2_pred = digit2_logits.argmax(dim=1, keepdim=True)

            # Count correct predictions overall
            # sum the instances where our prediction matches the target
            for pred1, pred2, target1, target2 in zip(digit1_pred, digit2_pred, digit1, digit2):
                if pred1 == target1:
                    correct += 0.5
                if pred2 == target2:
                    correct += 0.5
            
    test_loss = float(np.mean(losses))
    test_acc = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

    return test_loss, test_acc

def main(FLAGS):

    # Setup device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Using device:", device)

    # Call function to get dataloaders.
    train_DL, val_DL, test_DL= dataloader.get_double_MNIST("dataset/double_mnist", FLAGS)

    # Intialize Model
    model = Network(FLAGS.mode).to(device)

    # Define a loss function
    # Using Binary Cross Entropy
    criterion = nn.CrossEntropyLoss()

    # Define an optimizer function
    # Using Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # Intialize a best accuracy var
    best_accuracy = 0.0

    # Create lists to store loss and accuracy across training
    train_loss_list = []
    train_acc_list = []

    # the test lists will store both validation and testing results.
    val_loss_list = []
    val_acc_list = []

    # Run training for n_epochs as specified in config
    for epoch in range(1, FLAGS.num_epochs + 1):
        print('Epoch {:.0f}:'.format(epoch))
        train_loss, train_acc = train(model, device, train_DL,
                                      optimizer, criterion, epoch,
                                      FLAGS.batch_size)
        val_loss, val_acc = test(model, device, criterion, val_DL)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
    
    # Run a final test on the testing set.
    print('Performing final test...')

    test_loss, test_acc = test(model, device, criterion, test_DL)

    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}%")

    print_plots(train_loss_list, train_acc_list, val_loss_list, val_acc_list)

    print("Best recorded validation accuracy is {:2.2f}".format(best_accuracy))
    
    print("Training and evaluation finished.")

    return



if __name__ == '__main__':
    # Set parameters for Convolutional Neural Network
    parser = argparse.ArgumentParser('Multi-Label Image Classification Exercise')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-3')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=50,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)