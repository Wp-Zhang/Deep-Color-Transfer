import torch
import torch.nn as nn

# * Code adopted from https://gist.github.com/codeslake/a9f184783ce94769ec2595b5edfc9c97
class LearnableHistogram(nn.Module):
    def __init__(self, bin_num):
        super().__init__()
        self.bin_num = bin_num
        self.LHConv_1 = nn.Conv2d(
            1, bin_num, kernel_size=1, padding=0, stride=1, bias=True
        )
        self.relu = nn.ReLU(True)

        self.LHConv_1.bias.data = -torch.arange(0, 1, 1 / (self.bin_num))
        self.LHConv_1.weight.data = torch.ones(self.bin_num, 1, 1, 1)
        self.LHConv_1.bias.requires_grad = False
        self.LHConv_1.weight.requires_grad = False

    def forward(self, input):
        a1 = self.LHConv_1(input)
        a2 = torch.abs(a1)
        a3 = 1 - a2 * (self.bin_num - 1)
        a4 = self.relu(a3)
        return a4


def get_histogram2d(tensor: torch.Tensor, model: LearnableHistogram) -> torch.Tensor:
    """Get image histogram using a LearnableHistogram model

    Parameters
    ----------
    tensor : torch.Tensor
        Image tensor
    model : LearnableHistogram
        A learnable histogram model

    Returns
    -------
    torch.Tensor
        Histogram of given image tensor
    """
    # Preprocess
    hist_a = model((tensor[:, 1, :, :].unsqueeze(1) + 1) / 2)
    hist_b = model((tensor[:, 2, :, :].unsqueeze(1) + 1) / 2)

    # Network
    dim = hist_a.size()[2]
    bin = model.bin_num
    tensor1 = hist_a.repeat(1, bin, 1, 1)
    tensor2 = hist_b.repeat(1, 1, bin, 1).view(-1, bin * bin, dim, dim)

    pool = nn.AvgPool2d(dim)
    hist2d = pool(tensor1 * tensor2)
    hist2d = hist2d.view(-1, 1, bin, bin)

    return hist2d
