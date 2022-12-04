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
        "--config",
        default="configs/DeepColorTransfer.yaml",
        help="path to the configuration file",
    )
    parser.add_argument(
        "--weights",
        help="Staet dict path",
    )
    parser.add_argument(
        "--test_dir",
        help="Test data dir",
    )
    parser.add_argument(
        "--use_seg",
        action="store_true",
        help="Whether to use seg",
    )
    args = parser.parse_args()

    cfg = Box.from_yaml(open(args.config, "r").read())
    model_args = cfg.model_args
    optimizer_args = cfg.optimizer_args
    dataset_args = cfg.dataset_args
    trainer_args = cfg.trainer_args

    dm = TestDataModule(
        testset_dir=args.test_dir,
        l_bin=dataset_args.l_bin,
        ab_bin=dataset_args.ab_bin,
        num_classes=dataset_args.num_classes,
        use_seg=args.use_seg,
    )
    model = Model(
        l_bin=dataset_args.l_bin,
        ab_bin=dataset_args.ab_bin,
        num_classes=dataset_args.num_classes,
        **model_args,
        **optimizer_args,
    )
    model.model.load_state_dict(torch.load(args.weights))

    trainer = Trainer(accelerator="auto")

    out = trainer.predict(model, dm, return_predictions=True)
    for i, img in enumerate(out):
        img = Image.fromarray((img * 255).astype("uint8"))
        img.save(str(Path(args.test_dir) / f"{i}.jpg"))
