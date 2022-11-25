from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import cv2
from pathlib import Path
import numpy as np
from .preprocessing import get_histogram, get_common_seg_map, get_segwise_hist, one_hot


class Adobe5kDataset(Dataset):
    def __init__(
        self,
        dataset_info: pd.DataFrame,
        data_dir: str,
        l_bin: int,
        ab_bin: int,
        num_classes: int,
    ):
        super(Dataset, self).__init__()

        self.info = dataset_info

        self.data_dir = Path(data_dir)
        self.seg_dir = self.data_dir / "segs"
        self.in_img_dir = self.data_dir / "in_imgs"
        self.in_hist_dir = self.data_dir / "in_hist"
        self.ref_img_dir = self.data_dir / "ref_imgs"
        self.ref_hist_dir = self.data_dir / "ref_hist"

        self.img_transform = T.Compose(
            [
                T.ToTensor()
                # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes

    def __len__(self):
        return self.info.shape[0]

    def __getitem__(self, index):
        in_id = self.info["in_img"].iloc[index]
        ref_id = self.info["ref_img"].iloc[index]
        seg_id = self.info["seg"].iloc[index]

        in_img = np.load(str(self.in_img_dir / f"{in_id}.npy"))
        in_hist = np.load(str(self.in_hist_dir / f"{in_id}.npy"))
        ref_img = np.load(str(self.ref_img_dir / f"{ref_id}.npy"))
        ref_hist = np.load(str(self.ref_hist_dir / f"{ref_id}.npy"))
        seg = np.load(str(self.seg_dir / f"{seg_id}.npy"))

        in_common_seg = one_hot(seg, self.num_classes)
        ref_seg_hist = get_segwise_hist(
            ref_img.transpose(2, 0, 1),
            self.l_bin,
            self.ab_bin,
            seg,
            self.num_classes,
        )

        in_img = self.img_transform(in_img).float()
        ref_img = self.img_transform(ref_img).float()

        return in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_seg_hist


class TestDataset(Dataset):
    def __init__(self, data_dir: str, l_bin: int, ab_bin: int, num_classes: int):
        super(Dataset, self).__init__()

        self.data_dir = Path(data_dir)

        self.in_img_dir = self.data_dir / "in_imgs"
        self.in_seg_dir = self.data_dir / "in_segs"
        self.ref_seg_dir = self.data_dir / "ref_segs"
        self.ref_img_dir = self.data_dir / "ref_imgs"

        self.in_img_paths = sorted(list(self.in_img_dir.glob("**/*.jpg")))
        self.in_seg_paths = sorted(list(self.in_seg_dir.glob("**/*.npy")))
        self.ref_img_paths = sorted(list(self.ref_img_dir.glob("**/*.jpg")))
        self.ref_seg_paths = sorted(list(self.ref_seg_dir.glob("**/*.npy")))

        self.img_transform = T.Compose(
            [
                T.ToTensor(),
                # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes

    def __len__(self):
        return len(self.in_img_paths)

    def __getitem__(self, index):
        in_seg = np.load(str(self.in_seg_paths[index]))
        ref_seg = np.load(str(self.ref_seg_paths[index]))

        in_img = cv2.imread(str(self.in_img_paths[index]))
        in_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2LAB)
        ref_img = cv2.imread(str(self.ref_img_paths[index]))
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB)

        in_hist = get_histogram(in_img.transpose(2, 0, 1), self.l_bin, self.ab_bin)
        ref_hist = get_histogram(ref_img.transpose(2, 0, 1), self.l_bin, self.ab_bin)

        in_common_seg = get_common_seg_map(in_seg, ref_seg, self.num_classes)
        ref_seg_hist = get_segwise_hist(
            ref_img.transpose(2, 0, 1),
            self.l_bin,
            self.ab_bin,
            ref_seg,
            self.num_classes,
        )

        in_img = self.img_transform(in_img).float()
        ref_img = self.img_transform(ref_img).float()

        return in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_seg_hist
