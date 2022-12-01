# * Reference
# * Paper: Wang, Zhe, et al. "Learnable histogram: Statistical context features for deep
# *        neural networks." European Conference on Computer Vision. Springer, Cham, 2016.
# * Code:  https://www.kaggle.com/code/yerramvarun/learnable-histograms-implementation

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableHistogram(nn.Module):
    """Learnable histogram"""

    def __init__(self, in_channels: int, num_classes: int = 5, bin: int = 6):
        """Initialize a learnable histogram calculator.
        Number of output channels = num_classes * bins

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        num_classes : int, optional
            Number of classes, by default 5.
        bin : int, optional
            Number of histogram bins, by default 6.
        """
        super(LearnableHistogram, self).__init__()

        self.num_classes = num_classes
        self.bin = bin

        self.preconv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))

        # * Conv1
        self.conv1 = []
        for i in range(num_classes):
            # * filter for class i
            conv = nn.Conv2d(num_classes, bin, kernel_size=(1, 1))
            # * freeze weight
            conv.weight.requires_grad = False
            conv.weight.fill_(0)
            conv.weight[:, i, :, :] = 1

            self.conv1.append(conv)
        self.conv1 = nn.ModuleList(self.conv1)

        # * Conv2
        self.conv2 = nn.Conv2d(num_classes * bin, num_classes * bin, kernel_size=(1, 1))
        # * freeze bias
        self.conv2.bias.requires_grad = False
        self.conv2.bias.fill_(1)

    def forward(self, x):
        inp = self.preconv(x)

        # * Conv1, [(batch, num_classes, w, h)]
        tmp = [self.conv1[i](inp) for i in range(self.num_classes)]

        # * concat + abs
        concat = torch.cat(tmp, 1)  # * (batch, num_classes * bin, w, h)
        concat = torch.abs(concat)

        # * conv2
        out = self.conv2(concat)  # * (batch, num_classes * bin, w, h)

        out = F.relu(out)
        a, b = out.size()[:2]
        out = out.view(a, b, -1).mean(dim=2)  # * (batch, num_classes * bin)

        return out


# * Code : https://gist.github.com/codeslake/a9f184783ce94769ec2595b5edfc9c97
# class LearnableHistogram(nn.Module):
#     def __init__(self, bin_num):
#         super().__init__()
#         self.bin_num = bin_num
#         self.LHConv_1 = nn.Conv2d(
#             1, bin_num, kernel_size=1, padding=0, stride=1, bias=True
#         )
#         self.relu = nn.ReLU(True)

#         self.LHConv_1.bias.data = -torch.arange(0, 1, 1 / (self.bin_num))
#         self.LHConv_1.weight.data = torch.ones(self.bin_num, 1, 1, 1)

#     def forward(self, input):
#         a1 = self.LHConv_1(input)
#         a2 = torch.abs(a1)
#         a3 = 1 - a2 * (self.bin_num - 1)
#         a4 = self.relu(a3)
#         return a4


def get_histogram2d(tensor, model):
    # # Preprocess
    # hist_a = model((tensor[:, 1, :, :].unsqueeze(1) + 1) / 2)
    # hist_b = model((tensor[:, 2, :, :].unsqueeze(1) + 1) / 2)

    # # Network
    # dim = hist_a.size()[2]
    # bin = model.bin_num
    # tensor1 = hist_a.repeat(1, bin, 1, 1)
    # tensor2 = hist_b.repeat(1, 1, bin, 1).view(-1, bin * bin, dim, dim)

    # pool = nn.AvgPool2d(dim)
    # hist2d = pool(tensor1 * tensor2)
    # hist2d = hist2d.view(-1, 1, bin, bin)
    hist2d = model(tensor)

    return hist2d
