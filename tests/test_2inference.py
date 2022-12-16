from pytorch_lightning import Trainer
import torch
from box import Box
import warnings
import os

warnings.filterwarnings("ignore")

from src.data.preprocessing import get_dataset_info
from src.models import Model
from src.data import TestDataModule


def test_inference():
    get_dataset_info("tests/test_data/raw")

    # * Load config and dataset info
    cfg = Box.from_yaml(
        open("tests/test_configs/DeepColorTransfer-test.yaml", "r").read()
    )
    test_cfg = Box.from_yaml(open("tests/test_configs/test.yaml", "r").read())
    model_args = cfg.model_args
    optimizer_args = cfg.optimizer_args
    dataset_args = cfg.dataset_args

    lgt_model = Model(
        l_bin=dataset_args.l_bin,
        ab_bin=dataset_args.ab_bin,
        num_classes=dataset_args.num_classes,
        **model_args,
        **optimizer_args,
    )
    lgt_model.model.load_state_dict(torch.load(test_cfg.model_path))
    trainer = Trainer(accelerator=test_cfg.accelerator, devices=test_cfg.devices)

    # * Color transfer without semantic segmentation labels
    lgt_model.model.use_seg = False
    dm = TestDataModule(
        test_cfg.test_dir,
        dataset_args.l_bin,
        dataset_args.ab_bin,
        dataset_args.num_classes,
        False,
        test_cfg.resize_dim,
    )
    out = trainer.predict(lgt_model, dm)

    trainer = Trainer(accelerator="auto")

    # Color transfer with semantic segmentation labels
    lgt_model.model.use_seg = True
    dm.use_seg = True
    seg_out = trainer.predict(lgt_model, dm)

    assert len(out) == 1
    assert len(seg_out) == 1
    os.remove("tests/test_data/raw/dataset_info.csv")
