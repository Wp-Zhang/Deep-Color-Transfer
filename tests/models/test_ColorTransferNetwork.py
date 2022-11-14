from src.models import get_CTN
import torch


def test_CTN():
    model = get_CTN(3, 3, 64, use_dropout=True)

    img = torch.randn((2, 3, 128, 128))
    hist_enc1 = torch.randn((2, 64, 128, 128))
    hist_enc2 = torch.randn((2, 64, 128, 128))

    out = model(img, hist_enc1, hist_enc2)
