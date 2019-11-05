import sys
import torch
import torch.nn as nn
import torch.optim as optim

class Job(object):
    def __init__(self, model, loader_train, loader_test,
                 verbose=True):
        self.model = model
        self.loaders = {
            'train': loader_train,
            'test': loader_test
        }

        # Check for GPU acceleration
        self.device = torch.device("cuda:0"
                                   if torch.cuda.is_available()
                                   else "cpu")
        # Model settings
        self.verbose = verbose

    def train_model(self,
                    opt=optim.Adam,
                    criterion=nn.CrossEntropyLoss(),
                    epochs=3,
                    lr=0.0001):
        """
        """
        if not (self.loaders['train'] and self.loaders['test']):
            raise AttributeError('Data loaders have not been initialized.')

        optimizer = opt(self.model.parameters(), lr=lr)
        self.model.train()

        for i_epoch in range(epochs):
            running_loss = 0.0
            for i_data, data in enumerate(self.loaders['train']):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                if i_data % 1000 == 999 and self.verbose:
                    print('[%d, %5d] loss: %.3f' %
                    (i_epoch + 1, i_data + 1, running_loss / 1000))
                    sys.stdout.flush()
                    running_loss = 0.0

    def test_model(self):
        """
        """
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for data in self.loaders['test']:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: %d %%' % (acc))

def main():
    pass

if __name__ == '__main__':
    main()
