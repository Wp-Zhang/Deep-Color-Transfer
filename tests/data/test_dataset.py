from src.data import Adobe5kDataset, TestDataset
from src.data.preprocessing import get_dataset_info
import pandas as pd
import os


def test_adobe5k_dataset():
    get_dataset_info("tests/test_data/raw")
    info = pd.read_csv("tests/test_data/raw/dataset_info.csv")
    trainset = Adobe5kDataset(info, "tests/test_data/raw", (256, 256), 64, 64, 150)
    data = trainset[0]
    assert len(data) == 7
    _ = len(trainset)

    trainset = Adobe5kDataset(
        info, "tests/test_data/raw", (256, 256), 64, 64, 150, False
    )
    data = trainset[0]
    assert len(data) == 7
    _ = len(trainset)
    os.remove("tests/test_data/raw/dataset_info.csv")


def test_test_dataset():
    testset = TestDataset("tests/test_data/test", 64, 64, 150, True, 512)
    data = testset[0]
    assert len(data) == 6
    _ = len(testset)

    testset = TestDataset("tests/test_data/test", 64, 64, 150, False, 512)
    data = testset[0]
    assert len(data) == 6
    _ = len(testset)
