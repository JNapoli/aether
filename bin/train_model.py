"""MNIST training pipeline implemented in PyTorch.

This script implements a simple training pipeline for a convolutional neural
network model, intended for MNIST digit classification.

Details about the script's arguments can be displayed by running:
>>> python train_model.py -h

Dependencies for all packages and scripts can be installed using the
`requirements.txt` file in the repo's root directory.

This is intended for use as a processing script, and presently the data set
is hard-coded as MNIST, though this may be refactored at a later time so
that the user can easily swap in different datasets.
"""
import aether
import argparse
import numpy as np
import os
import torch

from torchvision import datasets, transforms


def main(args):
    # Get MNIST data loaders
    MNIST_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,], [0.5,])
    ])

    # Directory for saving dataset
    if not os.path.exists(args.path_save_datasets):
        os.mkdir(args.path_save_datasets)

    # Download training set and create dataloader
    train = datasets.MNIST(root=args.path_save_datasets,
                           train=True,
                           download=True,
                           transform=MNIST_transform)
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               num_workers=0)

    # Download test set and create dataloader
    test = datasets.MNIST(root=args.path_save_datasets,
                          train=False,
                          download=True,
                          transform=MNIST_transform)
    test_loader = torch.utils.data.DataLoader(test,
                                              batch_size=args.batch,
                                              shuffle=False,
                                              num_workers=0)

    # Create and train model
    model = aether.model.Convnet()

    # Create a job to bundle data with model
    job = aether.job.Job(model, train_loader, test_loader)

    # Train and save model
    job.train_model(epochs=args.epochs, lr=0.001, opt=torch.optim.SGD)
    job.test_model()
    torch.save(model.state_dict(), args.path_save_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run model training pipeline on the MNIST data set.'
    )
    parser.add_argument('path_save_model',
                        type=str,
                        help='Directory in which to save the model.')
    parser.add_argument('path_save_datasets',
                        type=str,
                        help='Directory in which to save the data sets.')
    parser.add_argument('path_save_data',
                        type=str,
                        help='Directory to save data output by the job.')
    parser.add_argument('-epochs',
                        type=int,
                        required=False,
                        default=5,
                        help='Number of training epochs.')
    parser.add_argument('-batch',
                        type=int,
                        required=False,
                        default=10,
                        help='Batch size to pass to data loaders.')
    args = parser.parse_args()
    main(args)
