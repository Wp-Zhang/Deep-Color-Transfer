from src.data import Adobe5kDataModule


def test_adobe5k_data_module():
    dm = Adobe5kDataModule("./data/processed", batch_size=2)

    dm.setup()
    trainset = dm.train_dataloader()
    validset = dm.val_dataloader()
    testset = dm.test_dataloader()
