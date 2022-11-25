from src.data import Adobe5kDataModule, TestDataModule


def test_adobe5k_data_module():
    dm = Adobe5kDataModule("tests/test_data/processed", 64, 64, 150, batch_size=2)

    dm.setup()
    trainset = dm.train_dataloader()
    validset = dm.val_dataloader()


def test_test_data_module():
    dm = TestDataModule("tests/test_data/test", 64, 64, 150, batch_size=2)
