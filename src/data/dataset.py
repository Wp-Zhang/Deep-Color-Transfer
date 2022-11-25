from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
from pathlib import Path
import numpy as np
from .preprocessing import preprocess_imgs


class Adobe5kDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        l_bin: int,
        ab_bin: int,
        num_classes: int,
        is_test: bool = False,
    ):
        super(Dataset, self).__init__()

        self.data_dir = Path(data_dir)

        self.in_img_dir = self.data_dir / "input" / "imgs"
        self.in_seg_dir = self.data_dir / "input" / "segs"
        self.in_hist_dir = self.data_dir / "input" / "hist"
        self.common_seg_dir = self.data_dir / "input" / "common_seg"
        self.ref_img_dir = self.data_dir / "reference" / "imgs"
        self.ref_seg_dir = self.data_dir / "reference" / "segs"
        self.ref_hist_dir = self.data_dir / "reference" / "hist"
        self.ref_seg_hist_dir = self.data_dir / "reference" / "seg_hist"

        self.in_img_paths = list(self.in_img_dir.glob("**/*.png"))

        self.img_transform = T.Compose(
            [
                T.ToTensor(),
                # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.seg_transform = T.Compose([T.ToTensor()])

        self.is_test = is_test
        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes

    def __len__(self):
        return len(self.in_img_paths)

    def __getitem__(self, index):
        no = self.in_img_paths[index].stem

        in_img = cv2.imread(str(self.in_img_paths[index]))
        ref_img = cv2.imread(str(self.ref_img_dir / f"{no}.png"))

        if self.is_test:
            in_seg = np.load(str(self.in_seg_dir / f"{no}.npy"))
            ref_seg = np.load(str(self.ref_seg_dir / f"{no}.npy"))
            processed = preprocess_imgs(
                in_img,
                in_seg,
                ref_img,
                ref_seg,
                None,
                self.l_bin,
                self.ab_bin,
                self.num_classes,
            )
            in_img = processed["in_img"]
            in_seg = processed["in_seg"]
            in_hist = processed["in_hist"]
            in_common_seg = processed["in_common_seg"]
            ref_img = processed["ref_img"]
            ref_seg = processed["ref_seg"]
            ref_hist = processed["ref_hist"]
            ref_seg_hist = ref_seg_hist["in_seg"]

        else:
            in_hist = np.load(str(self.in_hist_dir / f"{no}.npy"))
            in_common_seg = np.load(str(self.common_seg_dir / f"{no}.npy"))
            ref_hist = np.load(str(self.ref_hist_dir / f"{no}.npy"))
            ref_seg_hist = np.load(str(self.ref_seg_hist_dir / f"{no}.npy"))

        in_img = self.img_transform(in_img).float()
        ref_img = self.img_transform(ref_img).float()

        return in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_seg_hist
