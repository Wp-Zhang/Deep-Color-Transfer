import pytest
import torch
import numpy as np
import shutil
import random
from src.data.preprocessing import (
    get_histogram,
    get_segwise_hist,
    one_hot,
    gen_common_seg_map,
    preprocess_dataset,
)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # * Solve multiprocessing error


def test_get_histogram():
    h, w = 128, 256
    l_bin, ab_bin = 64, 64
    img = np.random.randint(0, 256, size=(3, h, w))
    hist = get_histogram(img, l_bin, ab_bin)
    assert hist.shape == (l_bin + 1, ab_bin, ab_bin)


def test_get_segwise_hist():
    h, w = 128, 256
    num_classes = 12
    l_bin, ab_bin = 64, 64
    seg = np.random.randint(0, num_classes - 1, size=(h, w))
    img = np.random.randint(0, 256, size=(3, h, w))
    hist = get_segwise_hist(img, l_bin, ab_bin, seg, num_classes)
    assert hist.shape == (num_classes, l_bin + 1, ab_bin, ab_bin)


def test_one_hot():
    seg = np.array([[1, 2, 3, 4], [0, 1, 7, 8]])
    res = one_hot(seg, num_classes=10)
    assert res.shape == (10, *seg.shape)


def test_gen_common_seg_map():
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
    res = gen_common_seg_map(seg1, seg2, 5)
    assert res.shape == (5, *seg1.shape)
    assert np.sum(expected - res, axis=None) == 0


def test_dataset_preprocessing():
    folder_name = random.randint(0, 10000)
    preprocess_dataset(
        raw_dir="./data/raw",
        processed_dir=f"./data/processed_{folder_name}/",
        resize_dim=(10, 10),
        l_bin=5,
        ab_bin=5,
        num_classes=50,
        n_jobs=1,
    )
    shutil.rmtree(f"./data/processed_{folder_name}")
