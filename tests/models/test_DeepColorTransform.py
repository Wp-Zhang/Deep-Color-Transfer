from src.models import DeepColorTransfer
import torch
from box import Box


def test_DeepColorTransfer():
    cfg_path = "tests/test_configs/DeepColorTransfer-test.yaml"
    cfg = Box.from_yaml(open(cfg_path, "r").read())
    model_args = cfg.model_args
    dataset_args = cfg.dataset_args

    model = DeepColorTransfer(
        l_bin=dataset_args.l_bin,
        ab_bin=dataset_args.ab_bin,
        num_classes=dataset_args.num_classes,
        **model_args,
    )

    b = 2
    l_bin = dataset_args.l_bin
    ab_bin = dataset_args.ab_bin
    h1, w1 = 65, 66
    h2, w2 = 67, 68
    num_seg_labels = 150

    in_img = torch.randn((b, 3, h1, w1))
    in_hist = torch.randn((b, l_bin + 1, ab_bin, ab_bin))
    in_common_seg = torch.randn((b, num_seg_labels, h1, w1)).round()
    ref_img = torch.randn((b, 3, h2, w2))
    ref_hist = torch.randn((b, l_bin + 1, ab_bin, ab_bin))
    ref_segwise_hist = torch.randn((b, num_seg_labels, l_bin + 1, ab_bin, ab_bin))

    x = in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
    out = model(*x)
