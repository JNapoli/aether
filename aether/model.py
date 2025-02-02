import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Convnet(nn.Module):
    """
    A class implementing a convolutional network in Pytorch for use with MNIST.

    Attributes
    ----------
    conv1 : torch.nn.Conv2d
        1st convolutional layer
    conv2 : torch.nn.Conv2d
        2nd convolutional layer
    conv3 : torch.nn.Conv2d
        3rd convolutional layer
    fc1 : torch.nn.Linear
        1st fully connected layer
    fc2 : torch.nn.Linear
        2nd fully connected layer

    Methods
    -------
    forward
        Required method of nn.Module subclasses that implements the forward pass
        and specifies activations / pooling.
    num_flat_features
        Helper method to get number of elements in tensor axes with index > 0
    """

    def __init__(self):
        super(Convnet, self).__init__()

        # 3 convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.conv3 = nn.Conv2d(16, 64, 1)

        # 2 fully-connected layers
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        """Performs the neural network's forward pass."""

        # Perform convolutions. Use Relu activations and max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        # Fully connected layers. First reshape x using view
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


    def num_flat_features(self, x):
        """Compute number of tensor elements in all axes except the first axis.
        """
        
        # Get all dimensions except batch dimension
        size = x.size()[1:]
        return np.prod(size)
