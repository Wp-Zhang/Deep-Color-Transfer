from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from pathlib import Path
import pandas as pd
from typing import Tuple
from .dataset import Adobe5kDataset, TestDataset


class Adobe5kDataModule(LightningDataModule):
    def __init__(
        self,
        trainset_dir: str,
        img_dim: Tuple[int, int],
        l_bin: int,
        ab_bin: int,
        num_classes: int,
        batch_size: int,
        num_workers: int = 8,
    ):
        super().__init__()

        self.trainset_dir = trainset_dir
        self.img_dim = img_dim
        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.info = pd.read_csv(str(Path(trainset_dir) / "dataset_info.csv"))

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            info = self.info[self.info["type"] == "train"].reset_index(drop=True)
            val_idx = list(range(0, info.shape[0], 5))
            train_info = info[~info.index.isin(val_idx)].reset_index(drop=True)
            val_info = info[info.index.isin(val_idx)].reset_index(drop=True)

            self.adb5k_train = Adobe5kDataset(
                train_info,
                self.trainset_dir,
                self.img_dim,
                self.l_bin,
                self.ab_bin,
                self.num_classes,
            )
            self.adb5k_val = Adobe5kDataset(
                val_info,
                self.trainset_dir,
                self.img_dim,
                self.l_bin,
                self.ab_bin,
                self.num_classes,
            )

        if stage == "validate":
            info = self.info[self.info["type"] == "train"].reset_index(drop=True)
            val_idx = list(range(0, info.shape[0], 5))
            val_info = info[info.index.isin(val_idx)].reset_index(drop=True)
            self.adb5k_val = Adobe5kDataset(
                val_info,
                self.trainset_dir,
                self.img_dim,
                self.l_bin,
                self.ab_bin,
                self.num_classes,
            )

        if stage == "fit" or stage == "validate" or stage is None:
            demo_info = self.info[self.info["type"] == "test"].reset_index(drop=True)
            self.adb5k_demo = Adobe5kDataset(
                demo_info,
                self.trainset_dir,
                self.img_dim,
                self.l_bin,
                self.ab_bin,
                self.num_classes,
            )

    def train_dataloader(self):
        return DataLoader(
            self.adb5k_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                self.adb5k_val,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.adb5k_demo,
                shuffle=False,
                batch_size=3,  # * same as # demo pictures
                num_workers=self.num_workers,
            ),
        ]


class TestDataModule(LightningDataModule):
    def __init__(
        self,
        testset_dir: str,
        l_bin: int,
        ab_bin: int,
        num_classes: int,
        use_seg: bool,
        batch_size: int,
        num_workers: int = 8,
    ):
        super().__init__()

        self.test_dir = testset_dir
        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes
        self.use_seg = use_seg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage) -> None:
        if stage == "predict" or stage is None:
            self.dataset = TestDataset(
                self.test_dir, self.l_bin, self.ab_bin, self.num_classes, self.use_seg
            )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
