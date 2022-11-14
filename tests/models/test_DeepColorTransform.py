from src.models import DCT
import torch
from box import Box


def test_DCT():
    cfg_path = "./configs/DeepColorTransfer.yaml"
    model = DCT(cfg_path)
    cfg = Box.from_yaml(open(cfg_path, "r").read())

    b = 2
    l_bin = cfg.l_bin
    ab_bin = cfg.ab_bin
    h1, w1 = 34, 34
    h2, w2 = 24, 24
    num_seg_labels = 50

    in_img = torch.randn((b, 3, h1, w1))
    in_hist = torch.randn((b, l_bin + 1, ab_bin, ab_bin))
    in_common_seg = torch.randn((b, num_seg_labels, h1, w1)).round()
    ref_img = torch.randn((b, 3, h2, w2))
    ref_hist = torch.randn((b, l_bin + 1, ab_bin, ab_bin))
    ref_segwise_hist = torch.randn((b, num_seg_labels, l_bin + 1, ab_bin, ab_bin))

    x = in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
    out = model(x)
