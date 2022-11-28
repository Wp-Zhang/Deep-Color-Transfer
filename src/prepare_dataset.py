from box import Box
import sys
import argparse

sys.path.append("./")

from src.data import get_dataset_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MIT-Adobe 5K dataset.")
    parser.add_argument(
        "--config",
        default="configs/DeepColorTransfer.yaml",
        help="path to the configuration file",
    )
    args = parser.parse_args()

    cfg = Box.from_yaml(open(args.config, "r").read())
    dataset_args = cfg.dataset_args
    get_dataset_info(dataset_args.raw_dir)
