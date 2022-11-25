from box import Box
import sys

sys.path.append("./")

from src.data import process_trainset

if __name__ == "__main__":
    cfg = Box.from_yaml(open("configs/DeepColorTransfer.yaml", "r").read())
    dataset_args = cfg.dataset_args
    process_trainset(
        dataset_args.raw_dir,
        dataset_args.processed_dir,
        dataset_args.resize_dim,
        dataset_args.l_bin,
        dataset_args.ab_bin,
        dataset_args.num_workers,
    )
