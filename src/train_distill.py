from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from box import Box
import sys
import warnings
import argparse
import torch

sys.path.append("./")
warnings.filterwarnings("ignore")

from src.models.distill_lightning_module import Model as DistillModel
from src.models import Model
from src.data import Adobe5kDataModule

if __name__ == "__main__":
    # * Load config and dataset info
    parser = argparse.ArgumentParser(
        description="Train DCT model on specified dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/DeepColorTransfer.yaml",
        help="path to the configuration file",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="W&B runner name",
    )
    args = parser.parse_args()

    cfg = Box.from_yaml(open(args.config, "r").read())
    model_args = cfg.model_args
    optimizer_args = cfg.optimizer_args
    dataset_args = cfg.dataset_args
    trainer_args = cfg.trainer_args

    cfg2 = Box.from_yaml(open("configs/DeepColorTransfer.yaml", "r").read())
    model_args2 = cfg2.model_args
    optimizer_args2 = cfg2.optimizer_args
    dataset_args2 = cfg2.dataset_args
    teacher_weights = torch.load("/root/autodl-tmp/weights.pt")

    dm = Adobe5kDataModule(
        trainset_dir=dataset_args.raw_dir,
        img_dim=dataset_args.img_dim,
        l_bin=dataset_args.l_bin,
        ab_bin=dataset_args.ab_bin,
        num_classes=dataset_args.num_classes,
        batch_size=dataset_args.batch_size,
        num_workers=dataset_args.num_workers,
    )
    teacher = Model(
        l_bin=dataset_args2.l_bin,
        ab_bin=dataset_args2.ab_bin,
        num_classes=dataset_args2.num_classes,
        **model_args2,
        **optimizer_args2,
    )
    teacher.load_state_dict(teacher_weights)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    model = DistillModel(
        teacher=teacher.model,
        l_bin=dataset_args.l_bin,
        ab_bin=dataset_args.ab_bin,
        num_classes=dataset_args.num_classes,
        **model_args,
        **optimizer_args,
    )
    model.model.HEN = teacher.model.HEN

    wandb_logger = WandbLogger(project="Deep Color Transform Distill", name=args.name)
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
