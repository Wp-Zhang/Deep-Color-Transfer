import torch
from pytorch_lightning import Trainer
from box import Box
import sys
import warnings

sys.path.append("./")
warnings.filterwarnings("ignore")

from src.models import DCT
from src.data import Adobe5kDataModule

if __name__ == "__main__":
    cfg = Box.from_yaml(open("configs/DeepColorTransfer.yaml", "r").read())
    model_args = cfg.model_args
    optimizer_args = cfg.optimizer_args
    dataset_args = cfg.dataset_args

    dm = Adobe5kDataModule(
        data_dir=dataset_args.processed_dir,
        batch_size=dataset_args.batch_size,
        num_workers=dataset_args.num_workers,
    )
    model = DCT(
        l_bin=dataset_args.l_bin,
        ab_bin=dataset_args.ab_bin,
        **model_args,
        **optimizer_args
    )

    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, dm)
