import torch
import torch.testing as tt
import torch.nn as nn

from src.util.init_weights import init_weights

import pytest


class Model(nn.Module):
    def __init__(self, c, h, w) -> None:
        super().__init__()

        self.conv = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c)
        self.linear = nn.Linear(h * w * 3, 128)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = out.view(x.size(0), -1)
        out = self.linear(out)


def test_init_weights():
    model = Model(3, 128, 128)
    init_weights(model, "Normal")
    init_weights(model, "Xavier")
    init_weights(model, "Kaiming")
    init_weights(model, "Orthogonal")

    with pytest.raises(NotImplementedError):
        init_weights(model, "XXX")
