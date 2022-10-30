import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
from pathlib import Path
import os
import numpy as np

from .preprocessing import get_histogram, get_segwise_hist, gen_common_seg_map


class Adobe5kDataset(Dataset):
    def __init__(self, data_dir, l_bin, ab_bin, num_classes):
        super(Dataset, self).__init__()

        self.data_dir = Path(data_dir)
        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes

        self.in_img_paths = list((self.data_dir / "input" / "imgs").glob("**/*.png"))
        self.in_img_segs = list((self.data_dir / "input" / "segs").glob("**/*.npy"))
        self.ref_img_paths = list(
            (self.data_dir / "reference" / "imgs").glob("**/*.png")
        )
        self.ref_img_segs = list(
            (self.data_dir / "reference" / "segs").glob("**/*.npy")
        )

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((128, 128)),  # ! TBD
                # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.in_img_paths)

    def __getitem__(self, index):
        in_img = cv2.imread(str(self.in_img_paths[index]))
        in_img = cv2.cvtColor(in_img, cv2.COLOR_RGB2LAB)
        # in_img = np.transpose(in_img, (2, 1, 0))
        in_img = self.transform(in_img).numpy()
        in_seg = np.load(self.in_img_segs[index])[0]

        ref_img = cv2.imread(str(self.ref_img_paths[index]))
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB)
        # ref_img = np.transpose(ref_img, (2, 1, 0))
        ref_img = self.transform(ref_img).numpy()
        ref_seg = np.load(self.ref_img_segs[index])[0]

        # in_img = self._rescale_img(in_img)
        # ref_img = self._rescale_img(ref_img)

        # * In case of mis-alignment
        in_seg = (
            F.upsample(
                torch.Tensor(in_seg[None, None, ...]),
                size=in_img.shape[1:],
                mode="bilinear",
            )
            .numpy()[0][0]
            .astype(int)
        )
        ref_seg = (
            F.upsample(
                torch.Tensor(ref_seg[None, None, ...]),
                size=ref_img.shape[1:],
                mode="bilinear",
            )
            .numpy()[0][0]
            .astype(int)
        )

        # ! tmp fix
        in_hist = np.random.rand(self.l_bin + 1, self.ab_bin, self.ab_bin)
        ref_hist = np.random.rand(self.l_bin + 1, self.ab_bin, self.ab_bin)
        ref_seg_hist = np.random.rand(
            self.num_classes, self.l_bin + 1, self.ab_bin, self.ab_bin
        )
        # in_hist = get_histogram(in_img, self.l_bin, self.ab_bin)
        # ref_hist = get_histogram(ref_img, self.l_bin, self.ab_bin)
        # ref_seg_hist = get_segwise_hist(
        #     ref_img, self.l_bin, self.ab_bin, ref_seg, self.num_classes
        # )

        in_common_seg = gen_common_seg_map(in_seg, ref_seg, self.num_classes)

        in_img = torch.from_numpy(in_img).float()
        in_hist = torch.from_numpy(in_hist).float()
        in_common_seg = torch.from_numpy(in_common_seg).long()
        ref_img = torch.from_numpy(ref_img).float()
        ref_hist = torch.from_numpy(ref_hist).float()
        ref_seg_hist = torch.from_numpy(ref_seg_hist).float()

        # print(
        #     in_img.size(),
        #     in_hist.size(),
        #     in_common_seg.size(),
        #     ref_img.size(),
        #     ref_hist.size(),
        #     ref_seg_hist.size(),
        # )
        return in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_seg_hist

    def _rescale_img(self, img, max_length=700):
        if (img.shape[1] > max_length) or (img.shape[2] > max_length):
            aspect_ratio = img.shape[1] / img.shape[2]
            if img.shape[1] > img.shape[2]:
                img = (
                    F.upsample(
                        torch.Tensor(img).unsqueeze(0),
                        size=(max_length, int(max_length / aspect_ratio)),
                        mode="bilinear",
                    )
                    .cpu()
                    .numpy()
                    .astype(int)[0]
                )
            else:
                img = (
                    F.upsample(
                        torch.Tensor(img).unsqueeze(0),
                        size=(int(max_length * aspect_ratio), max_length),
                        mode="bilinear",
                    )
                    .cpu()
                    .numpy()
                    .astype(int)[0]
                )
        return img
