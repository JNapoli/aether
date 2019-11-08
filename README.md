# Aether
This repository contains my Aether Biomachines take-home assignment. In this
repo, we implement a neural network training pipeline for MNIST handwritten
digit classification using [PyTorch](https://pytorch.org/).

## Overview

This repo contains scripts for training a convolutional neural
network on the MNIST data set and for launching an inference server that
serves the resulting model. The aether package contains
classes that simplify the training pipeline.

## Installation

The following steps have been verified to be reproducible on MacOS Catalina.
The code requires Python v3.7.5 and can be reproduced according to the following
steps:

1. Create and activate a Python environment ([Miniconda](https://docs.conda.io/en/latest/miniconda.html)
 is recommended) and install the dependencies in requirements-conda.txt:

```bash
conda create --name aether python=3.7.5 --file requirements-conda.txt
source activate aether
```

2. Clone this repository:
```bash
git clone https://github.com/JNapoli/aether.git
cd aether/
```

3. Source the env.sh file in the root directory. This will add the aether
package to your PYTHONPATH environment variable:
```bash
. env.sh
```

4. Confirm that the PYTHONPATH environment variable was properly set:
```bash
echo $PYTHONPATH
```

You should be good to go!

## Usage

### Re-fitting the model

A pre-trained model can be found in [./models/pretrained/](./models/pretrained/).
To fit a new model, run [train_model.py](./bin/train_model.py) in the bin directory:
```bash
python train_model.py /path/to/output/model.pt /path/for/storing/datasets/ \
       /path/to/directory/for/saving/output/data/ -epochs 10
```

For more details, you can run:
```bash
python train_model.py -h
```

The above will train the model over 10 epochs over the MNIST dataset
and output results / the model to the specified locations. It is recommended to
use full paths to filenames and directories. After training, the script will
print out the accuracy of the resulting model evaluated over the test set.
The pre-trained model included here was trained for 50 epochs and exhibits
98% test accuracy over the MNIST test set. This performance far exceeds the
random classification accuracy of 10% that would be expected if the model did
not learn anything.

### Serving the model and running inference

```bash
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
