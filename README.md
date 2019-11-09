# Aether
This repository contains my Aether Biomachines take-home assignment. In this
repo, we implement a neural network training pipeline for MNIST handwritten
digit classification using [PyTorch](https://pytorch.org/).

## Overview

This repo contains scripts for training a convolutional neural
network on the MNIST data set and for launching an inference server that
serves the resulting model. The aether package in this directory contains
classes that simplify the training pipeline.

## Installation

The following steps have been verified to be reproducible on MacOS Catalina.
The code requires Python v3.7.5 and can be reproduced according to the following
steps:

1. Clone this repository:
```bash
git clone https://github.com/JNapoli/aether.git
cd aether/
```

2. Create and activate a Python environment ([Miniconda](https://docs.conda.io/en/latest/miniconda.html)
 is recommended for this), providing the dependencies in requirements-conda.txt:

```bash
conda create --name aether python=3.7.5 --file requirements-conda.txt
source activate aether
```


3. Source the env.sh file in the root directory. This will add the aether
package to your PYTHONPATH environment variable:
```bash
source env.sh
```

4. Confirm that the PYTHONPATH environment variable was properly modified:
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

The above will train the model over 10 epochs on the MNIST dataset
and output results / the model to the specified locations. Please use
full paths for specifying files and directories. After training, the script will
print out the accuracy of the resulting model evaluated over the test set.

The pre-trained model included here was trained for 50 epochs on the torchvision MNIST
training dataset and scores 98% test accuracy on the MNIST test set. This
robust performance far exceeds the random classification accuracy of 10% that
would be expected if the model did not learn anything, and indicates the model
is not pathologically overfitting the training data.

### Serving the model
The resulting model can be served using the [Flask](https://flask.palletsprojects.com/en/1.1.x/)
app in [app.py](./bin/app.py), for example:

```bash
python app.py /repo_root_dir/models/pretrained/pretrained.pt
```

Running the above will launch the app and indicate where it is running (e.g. on my
machine, it indicates ```Running on http://127.0.0.1:5000/```).

### Inference
Inference may now be performed on additional data by making requests to the address
specified above. For example, the short python script:

```python
import requests

response = requests.post(
    'http://127.0.0.1:5000/predict',
    files={'file': open('./mnist_image.jpg', 'rb')}
)
print('Result:')
print(response.json()['class_name'])
print(response.json()['class_prob'])
```
Would run inference on the image ```mnist_image.jpg``` and print the result.
```response.json()``` returns a dictionary containing both the predicted
class of the number in the image as well as the probability associated with the
prediction.

In this repo, [run_inference.py](./bin/run_inference.py) is a script that runs
inference over a collection of jpg images contained in a directory. Running:

```bash
python run_inference.py /path/to/MNIST_jpeg_images/ http://127.0.0.1:5000/predict \
       /path/to/result.csv
```

will run inference for each jpg image contained in the provided directory and output
a csv file containing the image name, the predicted class, and the probability.
Inference was benchmarked on my machine and the rate was 96 images per second.
Further performance gains may be achieved by migrating to a more scalable production
server other than Flask.

For convenience, a [zipped directory](./data/MNIST_jpeg_for_inference.zip)
is included in this repo which contains MNIST images that can be used to test inference.
Inference has only been tested for jpg images, though support for other file types
may be added later.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
