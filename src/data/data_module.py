from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from pathlib import Path
from typing import Union
from .dataset import Adobe5kDataset


class Adobe5kDataModule(LightningDataModule):
    def __init__(
        self,
        trainset_dir: str,
        batch_size: int,
        num_workers: int = 8,
        testset_dir: Union[str, None] = None,
    ):
        super().__init__()

        self.trainset_dir = trainset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.testset_dir = testset_dir
        if testset_dir is not None:
            self.testset_dir = testset_dir

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            trainset = Adobe5kDataset(self.trainset_dir)
            self.adb5k_train, self.adb5k_val = random_split(trainset, [0.8, 0.2])

        if stage == "predict":
            if self.testset_dir:
                raise ValueError("Testset dir not defined")
            self.adb5k_pred = Adobe5kDataset(self.testset_dir)

    def train_dataloader(self):
        return DataLoader(
            self.adb5k_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.adb5k_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.adb5k_pred,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
