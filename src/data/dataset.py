import torch
from torch.utils.data import Dataset
from skimage import io, color
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

        self.in_img_paths = list((self.data_dir / "input" / "imgs").glob("**/*"))
        self.in_img_segs = list((self.data_dir / "input" / "segs").glob("**/*"))
        self.ref_img_paths = list((self.data_dir / "reference" / "imgs").glob("**/*"))
        self.ref_img_segs = list((self.data_dir / "reference" / "segs").glob("**/*"))

    def __len__(self):
        return len(self.in_img_paths)

    def __getitem__(self, index):
        in_img = color.rgb2lab(io.imread(self.in_img_paths[index]))
        in_seg = np.load(self.in_img_segs[index])
        ref_img = color.rgb2lab(io.imread(self.ref_img_paths[index]))
        ref_seg = np.load(self.ref_img_segs[index])

        in_hist = get_histogram(in_img, self.l_bin, self.ab_bin)
        ref_hist = get_histogram(in_seg, self.l_bin, self.ab_bin)
        ref_seg_hist = get_segwise_hist(
            ref_img, self.l_bin, self.ab_bin, ref_seg, self.num_classes
        )

        in_common_seg = gen_common_seg_map(in_seg, ref_seg, self.num_classes)

        in_img = torch.from_numpy(in_img)
        in_hist = torch.from_numpy(in_hist)
        in_common_seg = torch.from_numpy(in_common_seg)
        ref_img = torch.from_numpy(ref_img)
        ref_hist = torch.from_numpy(ref_hist)
        ref_seg_hist = torch.from_numpy(ref_seg_hist)

        return in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_seg_hist
