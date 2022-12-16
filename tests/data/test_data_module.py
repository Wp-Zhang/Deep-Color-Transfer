from src.data import Adobe5kDataModule, TestDataModule
from src.data.preprocessing import get_dataset_info
import os


def test_adobe5k_data_module():
    get_dataset_info(raw_dir="tests/test_data/raw")
    dm = Adobe5kDataModule("tests/test_data/raw", (256, 256), 64, 64, 150, batch_size=2)

    dm.setup("fit")
    dm.setup("validate")
    _ = dm.train_dataloader()
    _ = dm.val_dataloader()
    os.remove("tests/test_data/raw/dataset_info.csv")


def test_test_data_module():
    dm = TestDataModule("tests/test_data/test", 64, 64, 150, True, resize_dim=512)
    dm.setup("predict")
    _ = dm.predict_dataloader()
