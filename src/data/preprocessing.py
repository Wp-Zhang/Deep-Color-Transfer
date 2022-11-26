from typing import List, Tuple, Dict, Union
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm


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

    H = cv2.calcHist(
        [img.astype(np.float32)],
        channels=[1, 2],
        mask=None,
        histSize=[num_bin, num_bin],
        ranges=[0, 255, 0, 255],
    )
    H = H[None, ...]
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
    H = cv2.calcHist(
        [img.astype(np.float32)],
        channels=[0],
        mask=None,
        histSize=[num_bin],
        ranges=[0, 255],
    )
    H = H[..., None]
    H = H / np.sum(H, axis=None)

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


# * ===================================================


def _process_img(
    img_path: Path,
    seg_dir: Path,
    save_root: Path,
    resize_dim: Tuple[int, int],
    l_bin: int,
    ab_bin: int,
    expert_no: Union[str, None] = None,
):
    id = img_path.stem

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_NEAREST)
    hist = get_histogram(img.transpose(2, 0, 1), l_bin, ab_bin)
    seg = np.load(seg_dir / f"{id}.npy")
    seg = cv2.resize(seg, resize_dim, interpolation=cv2.INTER_NEAREST)

    postfix = f"_{expert_no}" if expert_no is not None else ""
    np.save(save_root / "imgs" / f"{id}{postfix}.npy", img.astype("uint8"))
    np.save(save_root / "hist" / f"{id}{postfix}.npy", hist)
    np.save(save_root / "hist" / f"{id}{postfix}.npy", hist)
    if expert_no is None:
        np.save(save_root / "segs" / f"{id}.npy", seg)


def process_trainset(
    raw_dir: str,
    processed_dir: str,
    resize_dim: Tuple[int, int],
    l_bin: int,
    ab_bin: int,
    n_jobs: int,
):
    """Preprocess the dataset and save the result.

    Parameters
    ----------
    raw_dir : str
        Path to raw dataset
    processed_dir : str
        Path to processed dataset
    l_bin : int
        Size of luminance bin
    ab_bin : int
        Size of ab bin
    num_classes: int
        Number of segmentation labels.
    n_jobs : int
        Number of jobs for parallel preprocessing
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    # * Prepare folders
    seg_dir = processed_dir / "segs"
    img_dir = processed_dir / "imgs"
    hist_dir = processed_dir / "hist"

    for dir in [seg_dir, img_dir, hist_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    # * Process input images
    img_paths = list((raw_dir / "adobe_5k" / "raw").glob("*.jpg"))
    parallel = Parallel(n_jobs=n_jobs, backend="multiprocessing")
    parallel(
        delayed(_process_img)(
            img_path,
            raw_dir / "segs",
            processed_dir,
            resize_dim,
            l_bin,
            ab_bin,
        )
        for img_path in tqdm(img_paths, desc=f"Processing input images")
    )

    # * Reference images
    for expert in ["a", "b", "c", "d", "e"]:
        img_paths = list((raw_dir / "adobe_5k" / expert).glob("*.jpg"))
        parallel = Parallel(n_jobs=n_jobs, backend="multiprocessing")
        parallel(
            delayed(_process_img)(
                img_path,
                raw_dir / "segs",
                processed_dir,
                resize_dim,
                l_bin,
                ab_bin,
                expert,
            )
            for img_path in tqdm(
                img_paths, desc=f"Processing reference images from expert {expert}"
            )
        )

    # * Generate dataset info
    info = None
    # * 5000*5 pairs
    raw_names = (raw_dir / "adobe_5k" / "raw").glob("*.jpg")
    raw_names = sorted([x.stem for x in raw_names])
    for expert in ["a", "b", "c", "d", "e"]:
        ref_names = (raw_dir / "adobe_5k" / expert).glob("*.jpg")
        ref_names = sorted([x.stem for x in ref_names])
        tmp = pd.DataFrame(columns=["in_img", "ref_img", "seg"])
        tmp["seg"] = raw_names
        tmp["in_img"] = raw_names
        tmp["ref_img"] = ref_names
        tmp["ref_img"] += f"_{expert}"
        info = pd.concat([info, tmp], ignore_index=True)

    # * identical pairs
    for expert in ["a", "b", "c", "d", "e"]:
        ref_names = (raw_dir / "adobe_5k" / expert).glob("*.jpg")
        ref_names = sorted([x.stem for x in ref_names])
        tmp = pd.DataFrame(columns=["in_img", "ref_img", "seg"])
        tmp["seg"] = ref_names
        tmp["in_img"] = ref_names
        tmp["ref_img"] = ref_names
        tmp["in_img"] += f"_{expert}"
        tmp["ref_img"] += f"_{expert}"
        info = pd.concat([info, tmp], ignore_index=True)
    info.to_csv(processed_dir / "dataset_info.csv", index=None)
