"""Helpers for dis-entangle"""

from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn.functional as F
from dis_entangle.data_loader_cache import im_preprocess, im_reader, normalize
from torch import nn
from torchvision import transforms


class GOSNormalize:
    """
    Normalize the Image using torch.transforms
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image

    def __repr__(self):
        return f"self.__class__.__name__:(mean={self.mean}, std={self.std})"


def load_image(im_path, hypar):
    """Load an image from a path and preprocess it."""
    transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])

    if im_path.startswith("http"):
        im_path = BytesIO(requests.get(im_path, timeout=60).content)

    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)  # make a batch of image, shape


def build_model(hypar, device):
    """Build the model."""
    net = hypar["model"]  # GOSNETINC(3,1)

    # convert to half precision
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if hypar["restore_model"] != "":
        net.load_state_dict(torch.load(hypar["model_path"] + "/" + hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()

    return net


def predict(net, inputs_val, shapes_val, hypar, device):
    """
    Given an Image, predict the mask
    """
    net.eval()

    if hypar["model_digit"] == "full":
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

    inputs_val_v = torch.Tensor(inputs_val, requires_grad=False).to(device)  # wrap inputs in Variable

    ds_val = net(inputs_val_v)[0]  # list of 6 results

    pred_val = ds_val[0][0, :, :, :]  # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(
        F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode="bilinear")
    )

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)  # max = 1

    if device == "cuda":
        torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)  # it is the mask we need
