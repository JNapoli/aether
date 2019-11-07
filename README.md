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
The code requires Python v3.7.5. It is recommended to first create and activate
a Python environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

```bash
conda create -n aether python=3.7.5
source activate aether
```

This package can then be downloaded and run as follows:

1. Clone this repo using:
```bash
git clone https://github.com/JNapoli/aether.git
cd aether/
```

2. Create a virtual environment (ensure that [Venv](https://docs.python.org/3.6/library/venv.html#module-venv) is available):
```bash
python3 -m venv my-env
source my-env/bin/activate
```

3. Install required packages via pip3:
```bash
pip3 install -r requirements.txt
```

## Usage

### Training

```bash
#python3 predict.py --path_track /FULL/PATH/TO/AUDIO/FILE/track.mp3
```

### Serving

```bash
#
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
