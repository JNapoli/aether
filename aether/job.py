import sys
import torch
import torch.nn as nn
import torch.optim as optim

class Job(object):
    """
    A class used to bundle train/test data together with the model to be fit.

    Attributes
    ----------
    model : torch.nn.Module
        Pytorch model to fit
    loaders : dict
        Contains data loaders for train/test data
    device :
        Checks whether GPUs are available
    verbose : bool
        Controls verbosity of the output

    Methods
    -------
    train_model
        Trains the model over the data yielded by the training loader
    test_model
        Tests the model over the data yielded by the test loader
    """

    def __init__(self, model, loader_train, loader_test, verbose=True):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Pytorch model to optimize
        loader_train : torch.utils.data.DataLoader
            Data loader for training data
        loader_test : torch.utils.data.DataLoader
            Data loader for testing data
        verbose : bool
            Controls verbosity of the output
        """

        self.model = model
        self.loaders = {
            'train': loader_train,
            'test': loader_test
        }

        # Check for GPU acceleration
        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")
        # Job settings
        self.verbose = verbose


    def train_model(self,
                    opt=optim.Adam,
                    criterion=nn.CrossEntropyLoss(),
                    epochs=3,
                    lr=0.0001,
                    stride_print=1000):
        """Train the model.

        Parameters
        ----------
        opt : Pytorch Optimizer object
            Optimizer to use
        criterion : Loss function
            Criterion to use for minimization
        epochs : int
            Number of epochs for training
        lr : float
            Learning rate to pass to the optimizer
        """

        if not (self.loaders['train'] and self.loaders['test']):
            raise AttributeError('Data loaders have not been initialized.')

        # Instantiate optimizer and set model to train mode
        optimizer = opt(self.model.parameters(), lr=lr)

        # Train and monitor loss
        # Note: Structure mirrors Pytorch tutorial @
        #   https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for i_epoch in range(epochs):

            self.model.train()
            running_loss = 0.0
            for i_data, data in enumerate(self.loaders['train']):

                # Evaluate outputs, perform backprop, and take a step
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Monitor progress
                if i_data % stride_print == 0 and self.verbose:
                    print('[%d, %5d] loss: %.3f' %
                    (i_epoch + 1, i_data + 1, running_loss / stride_print))
                    sys.stdout.flush()
                    running_loss = 0.0


    def test_model(self):
        """Evaluate the model over the test set and print accuracy."""

        # Set eval mode
        self.model.eval()

        # Accumulate stats
        total, correct = 0, 0
        with torch.no_grad():
            for data in self.loaders['test']:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print accuracy
        acc = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: %d %%' % (acc))
