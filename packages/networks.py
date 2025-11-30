import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, mode):
        super(Network, self).__init__()

        """ Define various Neural Network layers """

        # AlexNet Layers
        self.alex_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # dim: 64 -> 60 -> 30

        self.alex_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # dim: 30 -> 26 -> 13

        self.alex_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        ) # dim: 13 -> 11

        self.alex_conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        ) # dim: 11 -> 9

        self.alex_conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # dim: 9 -> 7 -> 3

        self.alex_fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*3*256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # ResNet Layers
        self.res_in_channels = 64
        self.res_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.res_layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.res_layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.res_layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.res_layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.res_avgpool = nn.AdaptiveAvgPool2d((1,1))

        # Fully Convolutional Layers
        self.custom_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # dim: 64 -> 60 -> 30

        self.custom_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # dim: 30 -> 26 -> 13

        self.custom_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        ) # dim: 13 -> 11

        self.custom_conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        ) # dim: 11 -> 9

        self.custom_conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # dim: 9 -> 7 -> 3
        
        self.fcn_layer = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        ) # dim: 3 -> 1

        # Two fully convolutional output heads for classification.
        self.fcn_output1 = nn.Conv2d(512, 10, kernel_size=1)
        self.fcn_output2 = nn.Conv2d(512, 10, kernel_size=1)

        # two output heads - used for all architectures
        self.fc_digit1 = nn.Linear(512, 10)
        self.fc_digit2 = nn.Linear(512, 10)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        else:
            print("Invalid mode ", mode, "selected. Select between 1-3")
            exit(0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """ Creates the sequential module for the layers of the residual network """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.res_in_channels, out_channels, stride))
            self.res_in_channels = out_channels

        # Create the Sequential layer module object by passing the pointer of the list of layers
        return nn.Sequential(*layers)

    # AlexNet-like architecture
    def model_1(self, X):

        # perform convolution layers
        X = self.alex_conv1(X)
        X = self.alex_conv2(X)
        X = self.alex_conv3(X)
        X = self.alex_conv4(X)
        X = self.alex_conv5(X)

        # move through classifier (flattens within the module)
        X = self.alex_fcs(X)

        # make two digit predictions
        out1 = self.fc_digit1(X)
        out2 = self.fc_digit2(X)

        return out1, out2
    
    # ResNet18-like architecture
    def model_2(self, X):

        # first convolution layer
        out = self.res_conv1(X)
        out = self.res_bn1(out)
        out = self.relu(out)

        # residual block layers
        out = self.res_layer1(out)
        out = self.res_layer2(out)
        out = self.res_layer3(out)
        out = self.res_layer4(out)

        # perform average pooling and flatten the tensor
        out = self.res_avgpool(out)
        out = out.view(out.size(0), -1)

        # classifier head - make two digit predictions
        out1 = self.fc_digit1(out)
        out2 = self.fc_digit2(out)

        return out1, out2
    
    # Custom architecture - Fully Convolutional Network
    def model_3(self, X):

        # perform convolution layers - reusing AlexNet architecture
        out = self.custom_conv1(X)
        out = self.custom_conv2(out)
        out = self.custom_conv3(out)
        out = self.custom_conv4(out)
        out = self.custom_conv5(out)

        # instead of using fully connected layers, we use fully convolutional layers here
        out = self.fcn_layer(out)

        # pass through classification heads
        out1 = self.fcn_output1(out)
        out2 = self.fcn_output2(out)

        # flatten to [batch, 10] for loss function to process
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)

        return out1, out2

# Class provided by Geek for Geeks
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # define a convolution layer w/ kernel size 3 and padding 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # define a batch normalization layer with out_channel parameters
        self.bn1 = nn.BatchNorm2d(out_channels)

        # define a ReLU layer where operation is done in-place
        self.relu = nn.ReLU(inplace=True)

        # define a second convolution layer with matching input and output channel sizes
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # define batch normalization layer with out_channels parameters
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    # define the forward pass function for the residual block.
    def forward(self, x):
        
        # two convolutional layers, each followed by BatchNorm and ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # shortcut connection adds input to output
        out += self.shortcut(x)
        out = self.relu(out)

        return out