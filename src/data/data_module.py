from torch.utils.data import DataLoader, random_split

# import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from pathlib import Path
from .dataset import Adobe5kDataset


class Adobe5kDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = None

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            trainset = Adobe5kDataset(str(self.data_dir))
            self.adb5k_train, self.adb5k_val = random_split(trainset, [0.8, 0.2])

        # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.adb5k_test = Adobe5kDataset(str(self.data_dir / "test"))

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

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.adb5k_test,
    #         shuffle=False,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #     )
