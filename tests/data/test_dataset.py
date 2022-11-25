from src.data import Adobe5kDataset, TestDataset
import pandas as pd


def test_adobe5k_dataset():
    info = pd.read_csv("tests/test_data/processed/dataset_info.csv")
    trainset = Adobe5kDataset(info, "tests/test_data/processed", 64, 64, 150)
    data = trainset[0]
    assert len(data) == 6
    size = len(trainset)


def test_test_dataset():
    testset = TestDataset("tests/test_data/test", 64, 64, 150)
    data = testset[0]
    assert len(data) == 6
    size = len(testset)
