from src.models.LearnableHistogram import LearnableHistogram, get_histogram2d
import torch


def test_LH():
    model = LearnableHistogram(3)
    b = 12
    img = torch.randn((b, 3, 32, 32))
    out = get_histogram2d(img, model)
