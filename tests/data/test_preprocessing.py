import os
import numpy as np
from src.data.preprocessing import (
    get_histogram,
    get_segwise_hist,
    one_hot,
    get_common_seg_map,
    get_dataset_info,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # * Solve multiprocessing error


def test_get_histogram():
    h, w = 128, 256
    l_bin, ab_bin = 64, 64
    img = np.random.randn(3, h, w)
    hist = get_histogram(img, l_bin, ab_bin)
    assert hist.shape == (l_bin + 1, ab_bin, ab_bin)


def test_get_segwise_hist():
    h, w = 128, 256
    num_classes = 12
    l_bin, ab_bin = 64, 64
    seg = np.random.randint(0, num_classes - 1, size=(h, w))
    img = np.random.randn(3, h, w)
    hist = get_segwise_hist(img, l_bin, ab_bin, seg, num_classes)
    assert hist.shape == (num_classes, l_bin + 1, ab_bin, ab_bin)


def test_one_hot():
    seg = np.array([[1, 2, 3, 4], [0, 1, 7, 8]])
    res = one_hot(seg, num_classes=10)
    assert res.shape == (10, *seg.shape)


def test_get_common_seg_map():
    seg1 = np.array([[1, 2, 3, 4], [0, 1, 4, 3]])
    seg2 = np.array([[0, 2, 3], [0, 0, 0]])

    expected = np.array(
        [
            [[0, 0, 0, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 0], [0, 0, 0, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    res = get_common_seg_map(seg1, seg2, 5)
    assert res.shape == (5, *seg1.shape)
    assert np.sum(expected - res, axis=None) == 0

    res2 = get_common_seg_map(seg2, seg2, 5)
    expected = one_hot(seg2, 5)
    assert np.sum(res2 - expected, axis=None) == 0


def test_dataset_preprocessing():
    get_dataset_info(raw_dir="tests/test_data/raw")
    os.remove("tests/test_data/raw/dataset_info.csv")
    # shutil.rmtree(f"./data/processed_{folder_name}")
