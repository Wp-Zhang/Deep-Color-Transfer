from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
from pathlib import Path
import numpy as np


class Adobe5kDataset(Dataset):
    def __init__(self, data_dir):
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

    def __len__(self):
        return len(self.in_img_paths)

    def __getitem__(self, index):
        no = self.in_img_paths[index].stem

        in_img = cv2.imread(str(self.in_img_paths[index]))
        in_img = self.img_transform(in_img).float()
        # in_seg = np.load(str(self.in_seg_dir / f"{no}.npy"))
        in_hist = np.load(str(self.in_hist_dir / f"{no}.npy"))
        in_common_seg = np.load(str(self.common_seg_dir / f"{no}.npy"))

        ref_img = cv2.imread(str(self.ref_img_dir / f"{no}.png"))
        ref_img = self.img_transform(ref_img).float()
        # ref_seg = np.load(str(self.ref_seg_dir / f"{no}.npy"))
        ref_hist = np.load(str(self.ref_hist_dir / f"{no}.npy"))
        ref_seg_hist = np.load(str(self.ref_seg_hist_dir / f"{no}.npy"))

        return in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_seg_hist
