from pathlib import Path
import pandas as pd
import numpy as np


def _get_ab_hist(img: "np.ndarray", num_bin: int) -> "np.ndarray":
    """Get ab-space histogram of an image
    Adopted from https://github.com/codeslake/Color_Transfer_Histogram_Analogy/blob/master/models/colorhistogram_model.py

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
    # Exclude Zeros and Make value 0 ~ 1
    arr1 = (img[1].ravel()[np.flatnonzero(img[1])] + 1) / 2
    arr2 = (img[2].ravel()[np.flatnonzero(img[2])] + 1) / 2

    if arr1.shape[0] != arr2.shape[0]:
        if arr2.shape[0] < arr1.shape[0]:
            arr2 = np.concatenate([arr2, np.array([0])])
        else:
            arr1 = np.concatenate([arr1, np.array([0])])

    # AB space
    arr_new = [arr1, arr2]
    H, _ = np.histogramdd(arr_new, bins=[num_bin, num_bin], range=((0, 1), (0, 1)))

    H = np.rot90(H)
    H = np.flip(H, 0)

    H = H[None, ...]
    H = H / np.sum(H)
    return H


def _get_l_hist(img: "np.ndarray", num_bin: int) -> "np.ndarray":
    """Get luminance histogram of an image
    Adopted from https://github.com/codeslake/Color_Transfer_Histogram_Analogy/blob/master/models/colorhistogram_model.py

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
    arr0 = (img[0].ravel()[np.flatnonzero(img[0])] + 1) / 2
    arr1 = np.zeros(arr0.size)

    arr_new = [arr0, arr1]
    H, _ = np.histogramdd(arr_new, bins=[num_bin, 1], range=((0, 1), (-1, 2)))
    H = H[None, ...].transpose(1, 0, 2)

    H = H / np.sum(H)

    return H


def get_histogram(img: "np.ndarray", l_bin: int, ab_bin: int) -> "np.ndarray":
    """Calculate image lab-space histogram

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
        if np.sum(seg == i) == 0:
            mask_hist = np.zeros((l_bin + 1, ab_bin, ab_bin))
        else:
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
    return (res == mask).astype("uint8")


def get_common_seg_map(
    input_seg: "np.ndarray[int]", ref_seg: "np.ndarray[int]", num_classes: int
) -> "np.ndarray[int]":
    """Get intersection of two segmentation maps

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
    input_oh = input_oh.astype("uint8")
    return input_oh


def get_dataset_info(raw_dir: str):
    """Generate dataset info and save as a .csv file.

    Parameters
    ----------
    raw_dir : str
        Path to raw dataset
    """
    raw_dir = Path(raw_dir)

    info = None

    # * 5000*5 pairs from original dataset
    raw_names = (raw_dir / "adobe_5k" / "raw").glob("*.jpg")
    raw_names = sorted([x.name for x in raw_names])
    seg_names = [x[:-3] + "npy" for x in raw_names]
    for expert in ["a", "b", "c", "d", "e"]:
        ref_names = (raw_dir / "adobe_5k" / expert).glob("*.jpg")
        ref_names = sorted([x.name for x in ref_names])
        tmp = pd.DataFrame()
        tmp["in_seg"] = seg_names
        tmp["in_img"] = raw_names
        tmp["in_img"] = "raw/" + tmp["in_img"]
        tmp["ref_img"] = ref_names
        tmp["ref_img"] = f"{expert}/" + tmp["ref_img"]
        tmp["trans"] = "Original"
        info = pd.concat([info, tmp], ignore_index=True)

    # * Identical pairs
    for expert in ["a", "b", "c", "d", "e"]:
        ref_names = (raw_dir / "adobe_5k" / expert).glob("*.jpg")
        ref_names = sorted([x.name for x in ref_names])
        tmp = pd.DataFrame()
        tmp["in_seg"] = seg_names
        tmp["in_img"] = ref_names
        tmp["ref_img"] = ref_names
        tmp["in_img"] = f"{expert}/" + tmp["in_img"]
        tmp["ref_img"] = f"{expert}/" + tmp["ref_img"]
        tmp["trans"] = "Identical"
        info = pd.concat([info, tmp], ignore_index=True)

    info["ref_seg"] = info["in_seg"]
    info["type"] = "train"

    # * Generate test demo for visualization during training process
    in_imgs = ["raw/a0029-jmac_DSC3974.jpg" for _ in range(4)]
    ref_imgs = [
        "c/a0029-jmac_DSC3974.jpg",
        "c/a0030-_MG_7844.jpg",
        "raw/a4966-Duggan_090124_4744.jpg",
        "raw/a0029-jmac_DSC3974.jpg",
    ]
    in_segs = ["a0029-jmac_DSC3974.npy" for _ in range(4)]
    ref_segs = [
        "a0029-jmac_DSC3974.npy",
        "a0030-_MG_7844.npy",
        "a4966-Duggan_090124_4744.npy",
        "a0029-jmac_DSC3974.npy",
    ]
    test_info = pd.DataFrame()
    test_info["in_seg"] = in_segs
    test_info["ref_seg"] = ref_segs
    test_info["in_img"] = in_imgs
    test_info["ref_img"] = ref_imgs
    test_info["trans"] = ["Original" for _ in range(3)] + ["HueShift"]
    test_info["type"] = "test"

    info = pd.concat([info, test_info], ignore_index=True)
    info = info[["in_img", "ref_img", "in_seg", "ref_seg", "trans", "type"]]
    info.to_csv(raw_dir / "dataset_info.csv", index=None)
