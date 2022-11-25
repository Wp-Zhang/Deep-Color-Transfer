from typing import Tuple, Dict, Union
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
        ranges=[0, 256, 0, 256],
    )
    H = H[None, ...]
    H = H / np.sum(H, axis=None)

    # arr = img.astype(float)

    # # Exclude Zeros and Make value 0 ~ 1
    # arr1 = (arr[1].ravel()[np.flatnonzero(arr[1])] + 1) / 2
    # arr2 = (arr[2].ravel()[np.flatnonzero(arr[2])] + 1) / 2

    # if arr1.shape[0] != arr2.shape[0]:
    #     if arr2.shape[0] < arr1.shape[0]:
    #         arr2 = np.concatenate([arr2, np.array([0])])
    #     else:
    #         arr1 = np.concatenate([arr1, np.array([0])])

    # # AB space
    # arr_new = [arr1, arr2]
    # H, edges = np.histogramdd(arr_new, bins=[num_bin, num_bin], range=((0, 1), (0, 1)))

    # H = np.rot90(H)
    # H = np.flip(H, 0)

    # H = H[None, ...].astype(float)
    # H = H / np.sum(H, axis=None)

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
        ranges=[0, 256],
    )
    H = H[..., None]
    H = H / np.sum(H, axis=None)

    return H

    # # Preprocess
    # arr = img.astype(float)
    # arr0 = (arr[0].ravel()[np.flatnonzero(arr[0])] + 1) / 2
    # arr1 = np.zeros(arr0.size)

    # arr_new = [arr0, arr1]
    # H, edges = np.histogramdd(arr_new, bins=[num_bin, 1], range=((0, 1), (-1, 2)))
    # H = np.transpose(H[None, ...], (1, 0, 2)).astype(float)

    # H = H / np.sum(H, axis=None)

    # return H


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
    return (res == mask).astype(int)


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


def preprocess_imgs(
    in_img: "np.ndarray[int]",
    in_seg: "np.ndarray[int]",
    ref_img: "np.ndarray[int]",
    ref_seg: "np.ndarray[int]",
    resize_dim: Union[Tuple[int, int], None],
    l_bin: int,
    ab_bin: int,
    num_classes: int,
) -> Dict:
    """Preprocess a single pair of images

    Parameters
    ----------
    in_img : np.ndarray[int]
        Input image
    in_seg : np.ndarray[int]
        Segmentation map of input image
    ref_img : np.ndarray[int]
        Reference image
    ref_seg : np.ndarray[int]
        Segmentation map of reference image
    resize_dim : Union[Tuple[int, int], None]
        Target resize dimension, (height, width)
    l_bin : int
        Size of luminance bin
    ab_bin : int
        Size of ab bin
    num_classes: int
        Number of segmentation labels.

    Returns
    -------
    D
        in_img, in_hist, in_common_seg, ref_img, \
            ref_hist, ref_seg_hist
    """
    # * Convert image color space
    in_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2LAB)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB)

    # * Resize
    if resize_dim is not None:
        in_img = cv2.resize(in_img, resize_dim, interpolation=cv2.INTER_NEAREST)
        in_seg = cv2.resize(in_seg, resize_dim, interpolation=cv2.INTER_NEAREST)
        ref_img = cv2.resize(ref_img, resize_dim, interpolation=cv2.INTER_NEAREST)
        ref_seg = cv2.resize(ref_seg, resize_dim, interpolation=cv2.INTER_NEAREST)

    # * Get histogram
    in_hist = get_histogram(in_img.transpose(2, 0, 1), l_bin, ab_bin)
    ref_hist = get_histogram(ref_img.transpose(2, 0, 1), l_bin, ab_bin)
    ref_seg_hist = get_segwise_hist(
        ref_img.transpose(2, 0, 1), l_bin, ab_bin, ref_seg, num_classes
    )

    in_common_seg = get_common_seg_map(in_seg, ref_seg, num_classes)

    return {
        "in_img": in_img,
        "in_seg": in_seg,
        "in_hist": in_hist,
        "in_common_seg": in_common_seg,
        "ref_img": ref_img,
        "ref_seg": ref_seg,
        "ref_hist": ref_hist,
        "ref_seg_hist": ref_seg_hist,
    }


def preprocess_single_pair(
    i,
    in_img_paths,
    in_seg_paths,
    ref_img_paths,
    ref_seg_paths,
    save_dir,
    resize_dim,
    l_bin,
    ab_bin,
    num_classes,
):
    in_img = cv2.imread(str(in_img_paths[i]))
    in_seg = np.load(in_seg_paths[i])[0]
    ref_img = cv2.imread(str(ref_img_paths[i]))
    ref_seg = np.load(ref_seg_paths[i])[0]

    res = preprocess_imgs(
        in_img,
        in_seg,
        ref_img,
        ref_seg,
        resize_dim,
        l_bin,
        ab_bin,
        num_classes,
    )

    p_in_img_dir = save_dir / "input" / "imgs"
    p_in_seg_dir = save_dir / "input" / "segs"
    p_in_hist_dir = save_dir / "input" / "hist"
    p_common_seg_dir = save_dir / "input" / "common_seg"
    p_ref_img_dir = save_dir / "reference" / "imgs"
    p_ref_seg_dir = save_dir / "reference" / "segs"
    p_ref_hist_dir = save_dir / "reference" / "hist"
    p_ref_seg_hist_dir = save_dir / "reference" / "seg_hist"

    cv2.imwrite(str(p_in_img_dir / f"{i}.png"), res["in_img"])
    np.save(p_in_seg_dir / f"{i}", res["in_seg"])
    np.save(p_in_hist_dir / f"{i}", res["in_hist"])
    np.save(p_common_seg_dir / f"{i}", res["in_common_seg"])

    cv2.imwrite(str(p_ref_img_dir / f"{i}.png"), res["ref_img"])
    np.save(p_ref_seg_dir / f"{i}", res["ref_seg"])
    np.save(p_ref_hist_dir / f"{i}", res["ref_hist"])
    np.save(p_ref_seg_hist_dir / f"{i}", res["ref_seg_hist"])


def preprocess_dataset(
    raw_dir: str,
    processed_dir: str,
    resize_dim: Tuple[int, int],
    l_bin: int,
    ab_bin: int,
    num_classes: int,
    n_jobs: int,
):
    """Preprocess the dataset and save the result.

    Parameters
    ----------
    raw_dir : str
        Path to raw dataset
    processed_dir : str
        Path to processed dataset
    resize_dim : Tuple[int, int]
        Target resize dimension, (width, height)
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

    for mode in ["train", "test"]:
        p_in_img_dir = processed_dir / mode / "input" / "imgs"
        p_in_seg_dir = processed_dir / mode / "input" / "segs"
        p_in_hist_dir = processed_dir / mode / "input" / "hist"
        p_common_seg_dir = processed_dir / mode / "input" / "common_seg"
        p_ref_img_dir = processed_dir / mode / "reference" / "imgs"
        p_ref_seg_dir = processed_dir / mode / "reference" / "segs"
        p_ref_hist_dir = processed_dir / mode / "reference" / "hist"
        p_ref_seg_hist_dir = processed_dir / mode / "reference" / "seg_hist"

        for dir in [
            p_in_img_dir,
            p_in_seg_dir,
            p_in_hist_dir,
            p_common_seg_dir,
            p_ref_img_dir,
            p_ref_seg_dir,
            p_ref_hist_dir,
            p_ref_seg_hist_dir,
        ]:
            dir.mkdir(parents=True, exist_ok=True)

        in_img_paths = sorted(
            list((raw_dir / mode / "input" / "imgs").glob("**/*.png"))
        )
        in_seg_paths = sorted(
            list((raw_dir / mode / "input" / "segs").glob("**/*.npy"))
        )
        ref_img_paths = sorted(
            list((raw_dir / mode / "reference" / "imgs").glob("**/*.png"))
        )
        ref_seg_paths = sorted(
            list((raw_dir / mode / "reference" / "segs").glob("**/*.npy"))
        )

        parallel = Parallel(n_jobs=n_jobs, backend="multiprocessing")
        parallel(
            delayed(preprocess_single_pair)(
                i,
                in_img_paths,
                in_seg_paths,
                ref_img_paths,
                ref_seg_paths,
                processed_dir / mode,
                resize_dim,
                l_bin,
                ab_bin,
                num_classes,
            )
            for i in tqdm(range(len(in_img_paths)), desc=f"Processing {mode} set")
        )
