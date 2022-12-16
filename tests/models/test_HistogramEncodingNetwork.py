from src.models import get_HEN
import torch


def test_HEN():
    model = get_HEN(65, 64)
    hist = torch.randn((12, 65, 64, 64))
    _ = model(hist)
