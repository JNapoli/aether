import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Convnet(nn.Module):
    def __init__(self, verbose=True):
        super(Convnet, self).__init__()

        # 3 convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)

        # 2 fully-connected layers
        self.fc1 = nn.Linear(32 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

        # Check for GPU acceleration
        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")

        # Model settings
        self.verbose = verbose
        self.loaders = {
            'train': None,
            'test': None
        }

    def forward(self, x):
        """
        """
        # Perform convolutions. Use Relu activations and max pooling.
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        # Fully connected layers. First reshape x using view.
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        """
        TODO: cite pytorch docs for this function
        """
        # Get all dimensions except batch dimension
        size = x.size()[1:]
        return np.prod(size)

    def set_loaders(self, loader_train, loader_test):
      """
      """
      self.loaders['train'] = loader_train
      self.loaders['test'] = loader_test
