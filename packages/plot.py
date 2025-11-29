import matplotlib.pyplot as plt
import numpy as np

def print_plots(train_loss, train_acc, test_loss, test_acc):

    # represents the number of data points we have, aka num of epochs trained for
    num_pts = len(train_loss) 

    line = np.linspace(1, num_pts, num_pts)

    # Training vs. Testing Loss figure
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Training vs. Testing Loss")
    ax1.plot(line, train_loss, c='blue')
    ax1.plot(line, test_loss,c='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(-0.25, 2.5)
    ax1.grid()
    fig1.savefig('loss.png')

    # Training vs. Testing Accuracy Figure
    fig2 = plt.figure()

    ax2 = fig2.add_subplot(111)
    ax2.set_title("Training vs. Testing Accuracy")
    ax2.plot(line, train_acc, c='blue')
    ax2.plot(line, test_acc, c='red')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 101)
    ax2.grid()
    fig2.savefig('acc.png')
    plt.show()

    return