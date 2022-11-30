from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from box import Box
import sys
import warnings
import argparse

sys.path.append("./")
warnings.filterwarnings("ignore")

from src.models import Model
from src.data import Adobe5kDataModule

if __name__ == "__main__":
    # * Load config and dataset info
    parser = argparse.ArgumentParser(description="Inference DCT model on demo dataset.")
    parser.add_argument(
        "--config",
        default="configs/DeepColorTransfer.yaml",
        help="path to the configuration file",
    )
    parser.add_argument(
        "--weights",
        help="Chekpoint path",
    )
    parser.add_argument(
        "--id",
        help="W&B runner id",
    )
    args = parser.parse_args()

    cfg = Box.from_yaml(open(args.config, "r").read())
    model_args = cfg.model_args
    optimizer_args = cfg.optimizer_args
    dataset_args = cfg.dataset_args
    trainer_args = cfg.trainer_args

    dm = Adobe5kDataModule(
        trainset_dir=dataset_args.raw_dir,
        img_dim=dataset_args.img_dim,
        l_bin=dataset_args.l_bin,
        ab_bin=dataset_args.ab_bin,
        num_classes=dataset_args.num_classes,
        batch_size=1,
        num_workers=dataset_args.num_workers,
    )
    model = Model(
        l_bin=dataset_args.l_bin,
        ab_bin=dataset_args.ab_bin,
        num_classes=dataset_args.num_classes,
        **model_args,
        **optimizer_args
    )

    wandb_logger = WandbLogger(project="Deep Color Transform", id=args.id)
    try:
        wandb_logger.experiment.config.update(cfg.to_dict())
    except:
        pass

    checkpoint_callback = ModelCheckpoint(
        dirpath=trainer_args.ckpt_dir, filename="DCT-{epoch:02d}-{val_loss:.4f}"
    )
    trainer = Trainer(
        accelerator=trainer_args.accelerator,
        devices=1,
        max_epochs=trainer_args.max_epochs,
        num_nodes=1,
        precision=trainer_args.precision,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )

    trainer.validate(model, dm, ckpt_path=args.weights)
