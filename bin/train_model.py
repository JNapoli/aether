import aether
import argparse
import os
import torch
import torchvision

from torchvision import datasets, transforms


def main(args):
    # Get MNIST data loaders
    MNIST_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,], [0.5,])
    ])

    # Directory for saving dataset
    if not os.path.exists(args.path_save_data):
        os.mkdir(args.path_save_data)

    # Download training set and create dataloader
    train = datasets.MNIST(root=args.path_save_data,
                           train=True,
                           download=True,
                           transform=MNIST_transform)
    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               num_workers=0)

    # Download test set and create dataloader
    test = torchvision.datasets.MNIST(root=args.path_save_data,
                                      train=False,
                                      download=True,
                                      transform=MNIST_transform)
    test_loader = torch.utils.data.DataLoader(test,
                                              batch_size=args.batch,
                                              shuffle=False,
                                              num_workers=0)
    # MNIST classes
    classes = np.arange(10)

    # Create and train model
    model = aether.model.Convnet()

    # Create a job to bundle data with model
    job = aether.job.Job(model, train_loader, test_loader)

    # Train and save model
    job.train_model(epochs=2, lr=0.001, opt=torch.optim.SGD)
    job.test_model()
    torch.save(model.state_dict(), args.path_save_model)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run model training pipeline on the MNIST data set.'
    )
    parser.add_argument('path_save_model',
                        type=str,
                        required=True,
                        help='Directory in which to save the model.')
    parser.add_argument('path_save_data',
                        type=str,
                        required=True,
                        help='Directory in which to save the data set.')
    parser.add_argument('-batch',
                        type=int,
                        required=False,
                        default=10,
                        help='Batch size to pass to data loaders.')
    args = parser.parse_args()
    main(args)
