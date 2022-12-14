import torch.nn as nn
from ..util import init_weights


class HistogramEncodingNetwork(nn.Module):
    """Histogram Encoding Network"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 128,
    ):
        """Initialize an HEN
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        hidden : int, optional
            Number of hidden channels, by default 128
        """
        super(HistogramEncodingNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden, hidden, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden, hidden, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden, hidden, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden, hidden, kernel_size=4, padding=1, stride=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden, hidden, kernel_size=4, padding=1, stride=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden, hidden, kernel_size=4, padding=1, stride=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(hidden, self.out_channels, kernel_size=1, padding=0),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(nn.Linear(out_channels, out_channels))

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.fc(x)  # (batch_size, 64)
        out = x[..., None, None]  # (batch_size, 64, 1, 1)

        return out


def get_HEN(
    in_channels: int,
    out_channels: int,
    hidden: int = 128,
    init_method="Normal",
) -> HistogramEncodingNetwork:
    """Construct and initialize weights of a HEN

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden : int, optional
        Number of hidden channels, by default 128.
    init_method : str, optional
        Weight initialization method, by default "Normal"

    Returns
    -------
    HistogramEncodingNetwork
        An HEN
    """
    model = HistogramEncodingNetwork(in_channels, out_channels, hidden)
    init_weights(model, method=init_method)

    return model
