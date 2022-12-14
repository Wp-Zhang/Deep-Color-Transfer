from pathlib import Path
from typing import Tuple, Union
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd
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
        """Adobe5k Data Module for training

        Parameters
        ----------
        trainset_dir : str
            Trainset directory
        img_dim : Tuple[int, int]
            Image size
        l_bin : int
            Bin number of l channel in lab color space
        ab_bin : int
            Bin number of a,b channels in lab color space
        num_classes : int
            Number of different semantic segmentation classes
        batch_size : int
            Batch size
        num_workers : int, optional
            Number of workers used for loading data, by default 8
        """
        super().__init__()

        self.trainset_dir = trainset_dir
        self.img_dim = img_dim
        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.info = pd.read_csv(str(Path(trainset_dir) / "dataset_info.csv"))

        self.adb5k_train = None
        self.adb5k_val = None
        self.adb5k_demo = None

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
                if_aug=False,
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
        resize_dim: Union[int, None],
    ):
        """Data module for model inference

        Parameters
        ----------
        testset_dir : str
            Test data directory
        l_bin : int
            Bin number of l channel in lab color space
        ab_bin : int
            Bin number of a,b channels in lab color space
        num_classes : int
            Number of different semantic segmentation classes
        use_seg : bool
            If input include semantic segmentation results
        resize_dim : Union[int, None]
            Scale the images into target size if specified
        """
        super().__init__()

        self.test_dir = testset_dir
        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes
        self.use_seg = use_seg
        if resize_dim is not None:
            resize_dim = int(resize_dim)
        self.resize_dim = resize_dim

        self.dataset = None

    def setup(self, stage=None):
        if stage == "predict" or stage is None:
            self.dataset = TestDataset(
                self.test_dir,
                self.l_bin,
                self.ab_bin,
                self.num_classes,
                self.use_seg,
                self.resize_dim,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=1,
            num_workers=0,
        )
