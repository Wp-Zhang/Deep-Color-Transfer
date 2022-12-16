import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import pandas as pd
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Tuple
from .preprocessing import (
    get_histogram,
    get_common_seg_map,
    get_segwise_hist,
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
        if_aug: bool = True,
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

        self.if_aug = if_aug

    def __len__(self):
        return self.info.shape[0]

    def augmentation(self, in_img, in_seg, ref_img, ref_seg, rand=True):
        # Resize
        resize = T.Resize(286, interpolation=T.InterpolationMode.NEAREST)
        in_img = resize(in_img)
        in_seg = resize(in_seg)
        ref_img = resize(ref_img)
        ref_seg = resize(ref_seg)

        if rand:
            # Random crop
            i, j, h, w = T.RandomCrop.get_params(in_img, output_size=self.img_dim)
            in_img = TF.crop(in_img, i, j, h, w)
            in_seg = TF.crop(in_seg, i, j, h, w)
            ref_img = TF.crop(ref_img, i, j, h, w)
            ref_seg = TF.crop(ref_seg, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                in_img = TF.hflip(in_img)
                in_seg = TF.hflip(in_seg)
                ref_img = TF.hflip(ref_img)
                ref_seg = TF.hflip(ref_seg)
        else:
            crop = T.CenterCrop(self.img_dim)
            in_img = crop(in_img)
            in_seg = crop(in_seg)
            ref_img = crop(ref_img)
            ref_seg = crop(ref_seg)

        return (
            in_img,
            np.array(in_seg).astype("uint8"),
            ref_img,
            np.array(ref_seg).astype("uint8"),
        )

    def __getitem__(self, index):
        in_name = self.info["in_img"].iloc[index]
        ref_name = self.info["ref_img"].iloc[index]
        in_seg_name = self.info["in_seg"].iloc[index]
        ref_seg_name = self.info["ref_seg"].iloc[index]
        trans = self.info["trans"].iloc[index]

        in_img = Image.open(str(self.img_dir / in_name)).convert("RGB")
        ref_img = Image.open(str(self.img_dir / ref_name)).convert("RGB")

        in_seg = Image.fromarray(
            np.load(str(self.seg_dir / in_seg_name), allow_pickle=True)
        )
        ref_seg = Image.fromarray(
            np.load(str(self.seg_dir / ref_seg_name), allow_pickle=True)
        )

        if self.if_aug:
            # * Trainset
            in_img, in_seg, ref_img, ref_seg = self.augmentation(
                in_img, in_seg, ref_img, ref_seg
            )
            if trans != "Identical":
                in_img = self.huelab_transform(in_img)
                ref_img = self.lab_transform(ref_img)
            else:
                in_img = self.huelab_transform(in_img)
                ref_img = in_img.clone()
        else:
            # * Demo for visualization
            in_img, in_seg, ref_img, ref_seg = self.augmentation(
                in_img, in_seg, ref_img, ref_seg, rand=False
            )
            if trans == "HueShift":
                in_img = self.lab_transform(in_img)
                ref_img = self.huelab_transform(ref_img)
            else:
                in_img = self.lab_transform(in_img)
                ref_img = self.lab_transform(ref_img)

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
            trans == "Identical",
        )


class TestDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        l_bin: int,
        ab_bin: int,
        num_classes: int,
        use_seg: bool,
        resize_dim: int = None,
    ):
        super(Dataset, self).__init__()

        self.data_dir = Path(data_dir)
        self.resize_dim = resize_dim

        self.in_img_dir = self.data_dir / "in_imgs"
        self.ref_img_dir = self.data_dir / "ref_imgs"
        self.in_img_paths = sorted(list(self.in_img_dir.glob("**/*.jpg")))
        self.ref_img_paths = sorted(list(self.ref_img_dir.glob("**/*.jpg")))

        self.in_seg_dir = self.data_dir / "in_segs"
        self.ref_seg_dir = self.data_dir / "ref_segs"
        self.in_seg_paths = sorted(list(self.in_seg_dir.glob("**/*.npy")))
        self.ref_seg_paths = sorted(list(self.ref_seg_dir.glob("**/*.npy")))

        self.lab_transform = get_transform_lab()

        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes

        self.use_seg = use_seg

    def __len__(self):
        return len(self.in_img_paths)

    def __getitem__(self, index):
        in_img = Image.open(str(self.in_img_paths[index])).convert("RGB")
        ref_img = Image.open(str(self.ref_img_paths[index])).convert("RGB")

        if self.use_seg:
            in_seg = Image.fromarray(
                np.load(str(self.in_seg_paths[index]), allow_pickle=True)
            )
            ref_seg = Image.fromarray(
                np.load(str(self.ref_seg_paths[index]), allow_pickle=True)
            )

            in_img = in_img.resize(in_seg.size)
            ref_img = ref_img.resize(ref_seg.size)

            if self.resize_dim is not None:
                resize = T.Resize(
                    self.resize_dim, interpolation=T.InterpolationMode.NEAREST
                )
                in_img = resize(in_img)
                ref_img = resize(ref_img)
                in_seg = resize(in_seg)
                ref_seg = resize(ref_seg)

            in_seg = torch.from_numpy(np.array(in_seg))
            ref_seg = torch.from_numpy(np.array(ref_seg))
        else:
            if self.resize_dim is not None:
                resize = T.Resize(self.resize_dim)
                in_img = resize(in_img)
                ref_img = resize(ref_img)
            in_seg = torch.from_numpy(np.array([0]))
            ref_seg = torch.from_numpy(np.array([0]))

        in_img = self.lab_transform(in_img).float()
        ref_img = self.lab_transform(ref_img).float()

        in_hist = get_histogram(in_img.numpy(), self.l_bin, self.ab_bin)
        ref_hist = get_histogram(ref_img.numpy(), self.l_bin, self.ab_bin)
        in_hist = torch.from_numpy(in_hist).float()
        ref_hist = torch.from_numpy(ref_hist).float()

        return in_img, in_hist, in_seg, ref_img, ref_hist, ref_seg
