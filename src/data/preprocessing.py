import numpy as np
import cv2


def _get_ab_hist(img: "np.ndarray", num_bin: int) -> "np.ndarray":
    """Get ab-space histogram of an image

    Parameters
    ----------
    img : np.ndarray
        Image numpy array
    num_bin : int
        Number of bins

    Returns
    -------
    np.ndarray
        Ab-space histogram
    """

    # H = cv2.calcHist(
    #     [img.astype(np.float32)],
    #     channels=[1, 2],
    #     mask=None,
    #     histSize=[num_bin, num_bin],
    #     ranges=[0, 256, 0, 256],
    # )
    # H = H[None, ...]
    # H = H / np.sum(H, axis=None)

    arr = img.astype(float)

    # Exclude Zeros and Make value 0 ~ 1
    arr1 = (arr[1].ravel()[np.flatnonzero(arr[1])] + 1) / 2
    arr2 = (arr[2].ravel()[np.flatnonzero(arr[2])] + 1) / 2

    if arr1.shape[0] != arr2.shape[0]:
        if arr2.shape[0] < arr1.shape[0]:
            arr2 = np.concatenate([arr2, np.array([0])])
        else:
            arr1 = np.concatenate([arr1, np.array([0])])

    # AB space
    arr_new = [arr1, arr2]
    H, edges = np.histogramdd(arr_new, bins=[num_bin, num_bin], range=((0, 1), (0, 1)))

    H = np.rot90(H)
    H = np.flip(H, 0)

    H = H[None, ...].astype(float)
    H = H / np.sum(H, axis=None)

    return H


def _get_l_hist(img: "np.ndarray", num_bin: int) -> "np.ndarray":
    """Get luminance histogram of an image

    Parameters
    ----------
    img : np.ndarray
        Image numpy array
    num_bin : int
        Number of bins

    Returns
    -------
    np.ndarray
        Luminance histogram
    """
    # H = cv2.calcHist(
    #     [img.astype(np.float32)],
    #     channels=[0, 1],
    #     mask=None,
    #     histSize=[num_bin, num_bin],
    #     ranges=[0, 256, 0, 256],
    # )
    # H = H[..., None]
    # H = H / np.sum(H, axis=None)

    # return H
    # Preprocess
    arr = img.astype(float)
    arr0 = (arr[0].ravel()[np.flatnonzero(arr[0])] + 1) / 2
    arr1 = np.zeros(arr0.size)

    arr_new = [arr0, arr1]
    H, edges = np.histogramdd(arr_new, bins=[num_bin, 1], range=((0, 1), (-1, 2)))
    H = np.transpose(H[None, ...], (1, 0, 2)).astype(float)

    H = H / np.sum(H, axis=None)

    return H


def get_histogram(img: "np.ndarray", l_bin: int, ab_bin: int) -> "np.ndarray":
    """_summary_

    Parameters
    ----------
    img : np.ndarray
        Image numpy array
    l_bin : int
        Size of luminance bin
    ab_bin : int
        Size of ab bin

    Returns
    -------
    np.ndarray
        Histogram
    """
    l_hist = _get_l_hist(img, l_bin)
    ab_hist = _get_ab_hist(img, ab_bin)

    l_hist = np.tile(l_hist, (1, ab_bin, ab_bin))

    hist = np.concatenate([ab_hist, l_hist], axis=0)

    return hist


def get_segwise_hist(
    img: "np.ndarray", l_bin: int, ab_bin: int, seg: "np.ndarray", num_classses: int
) -> "np.ndarray":
    """Get segmentation-wise histogram of an image

    Parameters
    ----------
    img : np.ndarray
        Image numpy array
    l_bin : int
        Size of luminance bin
    ab_bin : int
        Size of ab bin
    seg : np.ndarray
        Segementation map
    num_classses : int
        Number of segmentation labels

    Returns
    -------
    np.ndarray
        Histogram
    """
    l = []
    for i in range(num_classses):
        mask_img = img * (seg == i)
        mask_hist = get_histogram(mask_img, l_bin, ab_bin)
        l.append(mask_hist[None, :])

    return np.concatenate(l, axis=0)


def one_hot(seg: "np.ndarray[int]", num_classes: int) -> "np.ndarray[int]":
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
    input_seg: "np.ndarray[int]", ref_seg: "np.ndarray[int]", num_classes: int
) -> "np.ndarray[int]":
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
