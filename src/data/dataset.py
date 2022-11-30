import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import cv2
from PIL import Image
from skimage import color
from pathlib import Path
import numpy as np
from typing import Tuple
from .preprocessing import (
    get_histogram,
    get_common_seg_map,
    get_segwise_hist,
    resize_and_central_crop,
)
from .transforms import get_transform_lab, get_transform_hueshiftlab


class Adobe5kDataset(Dataset):
    def __init__(
        self,
        dataset_info: pd.DataFrame,
        data_dir: str,
        img_dim: Tuple[int, int],
        l_bin: int,
        ab_bin: int,
        num_classes: int,
    ):
        super(Dataset, self).__init__()

        self.info = dataset_info
        self.img_dim = img_dim

        self.data_dir = Path(data_dir)
        self.seg_dir = self.data_dir / "segs"
        self.img_dir = self.data_dir / "adobe_5k"

        self.lab_transform = get_transform_lab()
        self.huelab_transform = get_transform_hueshiftlab()

        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes

    def __len__(self):
        return self.info.shape[0]

    def __getitem__(self, index):
        in_name = self.info["in_img"].iloc[index]
        ref_name = self.info["ref_img"].iloc[index]
        in_seg_name = self.info["in_seg"].iloc[index]
        ref_seg_name = self.info["ref_seg"].iloc[index]
        trans = self.info["trans"].iloc[index]

        in_img = Image.open(str(self.img_dir / in_name)).convert("RGB")
        ref_img = Image.open(str(self.img_dir / ref_name)).convert("RGB")
        in_img = resize_and_central_crop(in_img, self.img_dim)
        ref_img = resize_and_central_crop(ref_img, self.img_dim)

        if trans == "Original":
            in_img = self.lab_transform(in_img)
            ref_img = self.lab_transform(ref_img)
        elif trans == "HueShift":
            ref_img = self.huelab_transform(in_img)
            in_img = self.lab_transform(in_img)

        in_seg = np.load(str(self.seg_dir / in_seg_name))
        in_seg = cv2.resize(in_seg, self.img_dim, interpolation=cv2.INTER_NEAREST)
        ref_seg = np.load(str(self.seg_dir / ref_seg_name))
        ref_seg = cv2.resize(ref_seg, self.img_dim, interpolation=cv2.INTER_NEAREST)
        in_hist = get_histogram(in_img.numpy(), self.l_bin, self.ab_bin)
        ref_hist = get_histogram(ref_img.numpy(), self.l_bin, self.ab_bin)

        in_common_seg = get_common_seg_map(in_seg, ref_seg, self.num_classes)
        ref_seg_hist = get_segwise_hist(
            ref_img.numpy(),
            self.l_bin,
            self.ab_bin,
            ref_seg,
            self.num_classes,
        )

        return (
            in_img.float(),
            torch.from_numpy(in_hist).float(),
            in_common_seg,
            ref_img.float(),
            torch.from_numpy(ref_hist).float(),
            torch.from_numpy(ref_seg_hist).float(),
        )


class TestDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        l_bin: int,
        ab_bin: int,
        num_classes: int,
        use_seg: bool,
        img_dim=(256, 256),
    ):
        super(Dataset, self).__init__()

        self.data_dir = Path(data_dir)
        self.use_seg = use_seg
        self.img_dim = img_dim

        self.in_img_dir = self.data_dir / "in_imgs"
        self.ref_img_dir = self.data_dir / "ref_imgs"
        self.in_img_paths = sorted(list(self.in_img_dir.glob("**/*.jpg")))
        self.ref_img_paths = sorted(list(self.ref_img_dir.glob("**/*.jpg")))

        if use_seg:
            self.in_seg_dir = self.data_dir / "in_segs"
            self.ref_seg_dir = self.data_dir / "ref_segs"
            self.in_seg_paths = sorted(list(self.in_seg_dir.glob("**/*.npy")))
            self.ref_seg_paths = sorted(list(self.ref_seg_dir.glob("**/*.npy")))

        self.lab_transform = get_transform_lab()

        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes

    def __len__(self):
        return len(self.in_img_paths)

    def __getitem__(self, index):
        in_img = Image.open(str(self.in_img_paths[index])).convert("RGB")
        ref_img = Image.open(str(self.ref_img_paths[index])).convert("RGB")

        in_img = resize_and_central_crop(in_img, self.img_dim)
        ref_img = resize_and_central_crop(ref_img, self.img_dim)

        in_img = self.lab_transform(in_img).float()
        ref_img = self.lab_transform(ref_img).float()

        in_hist = get_histogram(in_img.numpy(), self.l_bin, self.ab_bin)
        ref_hist = get_histogram(ref_img.numpy(), self.l_bin, self.ab_bin)

        if self.use_seg:
            in_seg = np.load(str(self.in_seg_paths[index]))
            ref_seg = np.load(str(self.ref_seg_paths[index]))

            in_seg = cv2.resize(in_seg, self.img_dim, interpolation=cv2.INTER_NEAREST)
            ref_seg = cv2.resize(ref_seg, self.img_dim, interpolation=cv2.INTER_NEAREST)

            in_common_seg = get_common_seg_map(in_seg, ref_seg, self.num_classes)
            ref_seg_hist = get_segwise_hist(
                ref_img.numpy(),
                self.l_bin,
                self.ab_bin,
                ref_seg,
                self.num_classes,
            )
        else:
            in_common_seg = np.array([])
            ref_seg_hist = np.array([])

        return (
            in_img,
            torch.from_numpy(in_hist).float(),
            in_common_seg,
            ref_img,
            torch.from_numpy(ref_hist).float(),
            torch.from_numpy(ref_seg_hist).float(),
        )
