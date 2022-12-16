from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from box import Box
import warnings
import os
import torch

warnings.filterwarnings("ignore")

from src.data.preprocessing import get_dataset_info
from src.models import Model
from src.data import Adobe5kDataModule


def test_train():
    get_dataset_info("tests/test_data/raw")
    cfg = Box.from_yaml(
        open("tests/test_configs/DeepColorTransfer-test.yaml", "r").read()
    )
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
        batch_size=dataset_args.batch_size,
        num_workers=dataset_args.num_workers,
    )
    model = Model(
        l_bin=dataset_args.l_bin,
        ab_bin=dataset_args.ab_bin,
        num_classes=dataset_args.num_classes,
        **model_args,
        **optimizer_args,
    )

    wandb_logger = WandbLogger(project="Deep Color Transform Test", name="test")
    try:
        wandb_logger.experiment.config.update(cfg.to_dict())
    except:
        pass

    checkpoint_callback = ModelCheckpoint(
        dirpath=trainer_args.ckpt_dir,
        filename=f"DCT-name={wandb_logger.experiment.name}-"
        + "epoch={epoch:02d}-val_loss={val_loss/dataloader_idx_0:.4f}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        accelerator=trainer_args.accelerator,
        devices=trainer_args.devices,
        max_epochs=trainer_args.max_epochs,
        sync_batchnorm=True,
        strategy="ddp_find_unused_parameters_false",
        num_nodes=1,
        precision=trainer_args.precision,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )

    trainer.fit(model, dm)
    os.remove("tests/test_data/raw/dataset_info.csv")

    # * save weights only
    m = Model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    torch.save(m.model.state_dict(), "tests/model.pt")
