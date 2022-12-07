# * Some code adopted from https://github.com/codeslake/Color_Transfer_Histogram_Analogy/blob/master/data/base_dataset.py

import torch
import torchvision.transforms as T
import numpy as np
from skimage import color
import random


A = 86.1830297444
B = 107.857300207
C = 98.2330538631 + 86.1830297444
D = 94.4781222765 + 107.857300207


def RGB2LAB(I):
    lab = color.rgb2lab(I)
    l = lab[:, :, 0] / 100.0
    a = (lab[:, :, 1] + A) / C
    b = (lab[:, :, 2] + B) / D
    return np.dstack([l, a, b])


def LAB2RGB(I):
    l = I[:, :, 0] / 255.0 * 100.0
    a = I[:, :, 1] / 255.0 * C - A
    b = I[:, :, 2] / 255.0 * D - B

    rgb = color.lab2rgb(np.dstack([l, a, b]).astype(np.float64))
    return rgb


def RGB2HSV_shift_LAB(I):
    shift, shift2 = random.random(), random.random() * 1.2
    shift2 = 0.3 if shift2 < 0.3 else shift2
    # Get Original L in LAB, shift H in HSV

    # Get Original LAB
    lab_original = color.rgb2lab(I)
    l_original = lab_original[:, :, 0]
    # Shift HSV
    hsv = color.rgb2hsv(I)
    h = hsv[:, :, 0] + shift
    s = (hsv[:, :, 1]) * shift2
    v = hsv[:, :, 2]
    hsv2 = color.hsv2rgb(np.dstack([h, s, v]).astype(np.float64))

    # Merge (Original LAB, Shifted HSV)
    lab = color.rgb2lab(hsv2)
    l = l_original / 100.0
    a = (lab[:, :, 1] + A) / C
    b = (lab[:, :, 2] + B) / D

    return np.dstack([l, a, b])


def get_transform_lab():
    """Image transformations for test data"""
    transform_list = [
        T.Lambda(lambda img: RGB2LAB(np.array(img))),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return T.Compose(transform_list)


def get_transform_hueshiftlab():
    """Image transformations for training data"""
    transform_list = [
        T.Lambda(lambda img: RGB2HSV_shift_LAB(np.array(img))),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return T.Compose(transform_list)


def post_process_img(img: torch.Tensor) -> "np.ndarray":
    """Turn the model output as an image

    Parameters
    ----------
    img : torch.Tensor
        Model output

    Returns
    -------
    np.ndarray
        Image array that can be directly showed
    """
    img = (img * 0.5 + 0.5) * 255
    img = img.cpu().numpy()
    img = LAB2RGB(img.transpose(1, 2, 0))

    return img
