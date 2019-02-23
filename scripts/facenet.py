import openface as op
import torch
import numpy as np


def forward(model, img):
    return model.forward(img.reshape(96,96,3))

def load_model(model_path):
    return op.TorchNeuralNet(model_path ,cuda=False)
