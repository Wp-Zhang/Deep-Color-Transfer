from src.data import Adobe5kDataset


def test_adobe5k_dataset():
    trainset = Adobe5kDataset("./data/processed/train")
    data = trainset[0]
    assert len(data) == 6
    size = len(trainset)
