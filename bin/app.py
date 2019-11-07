"""Create a server for inference over additional data using Flask.

NOTE: This server was adapted from the example in the Pytorch docs @
      https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html.
"""

import aether
import argparse
import io
import json
import os
import torch

from torchvision import models, transforms
from PIL import Image
from flask import Flask, jsonify, request


# Our Flask app
app = Flask(__name__)


# User can specify which model version to serve
parser = argparse.ArgumentParser(
        description='Serve the MNIST model using Flask.'
)
parser.add_argument('path_model',
                    type=str,
                    help='Full path to saved model (.pt file).')
args = parser.parse_args()


# Ensure the model exists
if not os.path.exists(args.path_model):
    raise FileNotFoundError('Desired model does not exist. Please generate.')
else:
    model = aether.model.Convnet()
    model.load_state_dict(torch.load(args.path_model))
    model.eval()


def transform_image(image_bytes):
    """Prepare a tensor to run inference.

    Parameters
    ----------
    image_bytes : Byte stream
        Raw image byte stream

    Returns
    -------
    Tensor object containing image data.
    """

    my_transforms = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5,], [0.5,])
    ])
    image = Image.open( io.BytesIO(image_bytes) )
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    """Run interence on input image.

    Parameters
    ----------
    image_bytes : Byte stream
        Raw image byte stream

    Returns
    -------
    y_prob : float
        Probability of the predicted class
    y_idx : int
        Predicted class
    """

    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    sm = torch.nn.Softmax(dim=1)
    outputs_sm = sm(outputs)
    y_prob, y_idx = torch.max(outputs_sm, 1)
    return y_prob.item(), y_idx.item()


@app.route('/predict', methods=['POST'])
def predict():
    """Process the request and return the result.

    Returns
    -------
    flask.Response object containing the result
    """

    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_prob, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_prob': class_prob,
                        'class_name': class_name})
    else:
        raise NotImplementedError('Can only handle POST requests.')


if __name__ == '__main__':
    app.run()
