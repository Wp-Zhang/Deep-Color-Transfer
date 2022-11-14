from src.models import LearnableHistogram
import torch


def test_LH():
    model = LearnableHistogram(3)
    b = 12
    img = torch.randn((b, 3, 32, 32))
    out = model(img)
