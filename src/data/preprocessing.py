from typing import Tuple
import torch
import numpy as np


# def one_hot(seg: torch.Tensor, num_classes: int) -> torch.Tensor:
#     """One-hot encode segmentation map

#     Parameters
#     ----------
#     seg : torch.Tensor
#         Segmentation map
#     num_classes : int
#         Number of segmentation labels

#     Returns
#     -------
#     torch.Tensor
#         One-hot encoded segmentation map with shape of (batch, num_classes, w, h)
#     """
#     b, w, h = seg.size()
#     index = seg.unsqueeze(1).type(torch.int64)  # (batch, 1, w, h)
#     res = torch.zeros(b, num_classes, w, h, dtype=seg.dtype)

#     return res.scatter(1, index, 1)


def one_hot(seg: np.ndarray[int], num_classes: int) -> np.ndarray[int]:
    """One-hot encode segmentation map

    Parameters
    ----------
    seg : np.ndarray[int]
        Segmentation map
    num_classes : int
        Number of segmentation labels

    Returns
    -------
    np.ndarray[int]
        One-hot encoded segmentation map with shape of (num_classes, w, h)
    """
    w, h = seg.shape
    res = np.tile(seg[None, ...], (num_classes, 1, 1))
    mask = np.ones((num_classes, w, h)) * np.arange(num_classes)[..., None, None]
    return (res == mask).astype(int)


def gen_common_seg_map(
    input_seg: np.ndarray[int], ref_seg: np.ndarray[int], num_classes: int
) -> np.ndarray[int]:
    """_summary_

    Parameters
    ----------
    input_seg : np.ndarray[int]
        Segmentation label of input image.
    ref_seg : np.ndarray[int]
        Segmentation label of reference image.
    num_classes : int
        Number of segmentation labels.

    Returns
    -------
    np.ndarray[int]
        One-hot encoded input img seg map, only preserve common seg labels
    """
    in_uni = np.unique(input_seg)
    ref_uni = np.unique(ref_seg)
    common = np.intersect1d(in_uni, ref_uni)  # * common segmentation labels

    input_oh = one_hot(input_seg, num_classes)  # (num_labels, w1, h1)
    input_oh[~np.isin(np.arange(num_classes), common), :, :] = 0

    return input_oh
