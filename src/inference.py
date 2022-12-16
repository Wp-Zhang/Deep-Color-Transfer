from pytorch_lightning import Trainer
import torch
from PIL import Image
from box import Box
from pathlib import Path
import sys
import warnings
import argparse

sys.path.append("./")
warnings.filterwarnings("ignore")

from src.models import Model
from src.data import TestDataModule

if __name__ == "__main__":
    # * Load config and dataset info
    parser = argparse.ArgumentParser(description="Inference DCT model on demo dataset.")
    parser.add_argument(
        "--model_config",
        default="configs/DeepColorTransfer.yaml",
        help="path to the model config file",
    )
    parser.add_argument(
        "--test_config",
        default="configs/test.yaml",
        help="path to the test config file",
    )
    args = parser.parse_args()

    cfg = Box.from_yaml(open(args.model_config, "r").read())
    test_cfg = Box.from_yaml(open(args.test_config, "r").read())
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

    dm = TestDataModule(
        test_cfg.test_dir,
        dataset_args.l_bin,
        dataset_args.ab_bin,
        dataset_args.num_classes,
        test_cfg.use_seg,
        test_cfg.resize_dim,
    )

    out = trainer.predict(lgt_model, dm, return_predictions=True)
    for i, img in enumerate(out):
        img = Image.fromarray((img * 255).astype("uint8"))
        img.save(str(Path(test_cfg.test_dir) / f"{i}.jpg"))
