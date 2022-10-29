from torch.utils.data import DataLoader, random_split

# import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from pathlib import Path
from .dataset import Adobe5kDataset


class Adobe5kDataModule(LightningDataModule):
    def __init__(
        self, data_dir: str, l_bin: int, ab_bin: int, num_classes: int, batch_size: int
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.transform = None

    # def prepare_data(self):
    #     pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            trainset = Adobe5kDataset(
                str(self.data_dir / "train"), self.l_bin, self.ab_bin, self.num_classes
            )
            self.adb5k_train, self.adb5k_val = random_split(trainset, [0.8, 0.2])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.adb5k_test = Adobe5kDataset(
                str(self.data_dir / "test"), self.l_bin, self.ab_bin, self.num_classes
            )

    def train_dataloader(self):
        return DataLoader(self.adb5k_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.adb5k_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.adb5k_test, batch_size=self.batch_size)
