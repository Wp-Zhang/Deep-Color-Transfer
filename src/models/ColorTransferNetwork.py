from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from ..util import init_weights

# * ============================== Modules ==============================


class UNetEncoder(nn.Module):
    """U-Net Encoder"""

    def __init__(self, in_channels: int, hidden: int):
        """Initialize a U-Net encoder

        Parameters
        ----------
        in_channels : int
            Number of input channels
        hidden : int
            Number of hidden channels
        """
        super(UNetEncoder, self).__init__()

        self.enc1 = self.build_block(in_channels, hidden, is_first=True)
        self.enc2 = self.build_block(hidden * 1, hidden * 2, False)  # 64, 128
        self.enc3 = self.build_block(hidden * 2, hidden * 4, False)  # 128, 256
        self.enc4 = self.build_block(hidden * 4, hidden * 8, False)  # 256, 512
        self.enc5 = self.build_block(hidden * 8, hidden * 8, False)  # 512, 512

    def build_block(self, in_channels, out_channels, is_first):
        if is_first:
            layers = []
        else:
            layers = [nn.LeakyReLU(0.2, True)]

        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
        ]

        return nn.Sequential(*layers)

    def forward(self, input):
        out1 = self.enc1.forward(input)  # 256/256/64 -> 128/128/64
        out2 = self.enc2.forward(out1)  # 128/128/64 -> 64/64/128
        out3 = self.enc3.forward(out2)  # 64/64/128 -> 32/32/256
        out4 = self.enc4.forward(out3)  # 32/32/256 -> 16/16/512
        out5 = self.enc5.forward(out4)  # 16/16/512 -> 8/8/512

        return out1, out2, out3, out4, out5


class UNetDecoderBlock(nn.Module):
    """UNet Decoder Block"""

    def __init__(self, channel_list: List):
        """Initialize a U-Net decoder block

        Parameters
        ----------
        channel_list : List
            List of number of channels for each conv layer, length should be 4.
        """
        super(UNetDecoderBlock, self).__init__()

        assert len(channel_list) == 4, "Incorrect length of channel_list"

        self.block = nn.Sequential(
            [
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(
                    channel_list[0],
                    channel_list[1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.InstanceNorm2d(channel_list[1]),
                # ------------------------------------------------
                nn.ReLU(True),
                nn.Conv2d(
                    channel_list[1], channel_list[2], kernel_size=3, stride=1, padding=1
                ),
                nn.InstanceNorm2d(channel_list[2]),
                # ------------------------------------------------
                nn.ReLU(True),
                nn.Conv2d(
                    channel_list[2], channel_list[3], kernel_size=3, stride=1, padding=1
                ),
                nn.InstanceNorm2d(channel_list[3]),
            ]
        )

    def forward(
        self,
        enc1: torch.Tensor,
        enc2: torch.Tensor,
        hist_enc1: torch.Tensor,
        hist_enc2: torch.Tensor,
        target_size1: int,
        target_size2: int,
    ) -> torch.Tensor:
        unsample_size = (target_size1, target_size2)

        enc1 = F.upsample(enc1, size=unsample_size, mode="bilinear")
        enc2 = F.upsample(enc2, size=unsample_size, mode="bilinear")
        hist_enc1 = F.upsample(hist_enc1, size=unsample_size, mode="bilinear")
        hist_enc2 = F.upsample(hist_enc2, size=unsample_size, mode="bilinear")

        out = self.block(torch.cat([enc1, enc2, hist_enc1, hist_enc2], 1))
        return out


class UNetDecoder(nn.Module):
    """U-Net Decoder"""

    def __init__(self, in_channels):
        super(UNetDecoder, self).__init__()

        self.dec1 = UNetDecoderBlock([in_channels, 512, 512, 512])
        self.dec2 = UNetDecoderBlock([in_channels, 256, 256, 256])
        self.dec3 = UNetDecoderBlock([in_channels // 2, 128, 128, 128])
        self.dec4 = UNetDecoderBlock([in_channels // 4, 64, 64, 64])
        self.dec5 = UNetDecoderBlock([in_channels // 8, 128, 64, 64])

    def forward(self, enc1, enc2, enc3, enc4, enc5, hist_enc1, hist_enc2, input_img):
        out1 = self.dec1(enc5, enc5, hist_enc1, hist_enc2, enc5.size(2), enc5.size(3))
        out2 = self.dec2(out1, enc4, hist_enc1, hist_enc2, enc4.size(2), enc4.size(3))
        out3 = self.dec3(out2, enc3, hist_enc1, hist_enc2, enc3.size(2), enc3.size(3))
        out4 = self.dec4(out3, enc2, hist_enc1, hist_enc2, enc2.size(2), enc2.size(3))
        out5 = self.dec5(out4, enc1, hist_enc1, hist_enc2, enc1.size(2), enc1.size(3))
        out5 = F.upsample(
            out5, size=(input_img.size(2), input_img.size(3)), mode="bilinear"
        )

        return out1, out2, out3, out4, out5


class RefinementBlock(nn.Module):
    """Refinement module block"""

    def __init__(self, in_channels: int, use_dropout=False):
        """Initialize a refinement block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        use_dropout : bool, optional
            If use dropout, by default False
        """
        super(RefinementBlock, self).__init__()
        self.block = self.build_block(in_channels, use_dropout)

    def build_block(self, in_channels: int, use_dropout: bool) -> nn.Module:
        """Build block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        use_dropout : bool
            If use dropout.

        Returns
        -------
        nn.Module
            A refinement block
        """
        layers = [
            nn.Conv2d(
                in_channels * 3, in_channels * 3, kernel_size=3, padding=1, bias=True
            ),
            nn.InstanceNorm2d(in_channels * 3),
            nn.ReLU(True),
        ]

        if use_dropout:
            layers += [nn.Dropout(0.5)]

        layers += [
            nn.Conv2d(
                in_channels * 3, in_channels, kernel_size=3, padding=1, bias=True
            ),
            nn.InstanceNorm2d(in_channels),
        ]

        return nn.Sequential(*layers)

    def forward(self, x, hist_enc1, hist_enc2):  # 64/64/64
        hist_enc1 = F.upsample(hist_enc1, size=(x.size(2), x.size(3)), mode="bilinear")
        hist_enc2 = F.upsample(hist_enc2, size=(x.size(2), x.size(3)), mode="bilinear")

        x_cat = torch.cat((x, hist_enc1, hist_enc2), 1)  # 64/64/64/
        out = x + self.block(x_cat)  # 192 -> 64
        return out


# * ======================= ColorTransferNetwork =======================


class ColorTransferNetwork(nn.Module):
    """Color Transfer Network"""

    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False):
        """Initialize a CTN
        Parameters
        ----------
        input_nc : int
            Number of input channels
        output_nc : int
            Number of output channels
        use_dropout : bool, optional
            If use dropout, by default False
        """
        super(ColorTransferNetwork, self).__init__()

        enc_nc = 64

        # * Pre-refienment block
        self.pre_refine_block = self.build_pre_refine_block(in_channels, enc_nc)

        # * Encoder
        self.UNetEnc = UNetEncoder(in_channels, 64)

        # * Decoder
        self.UNetDec = UNetDecoder(512 + 512 + enc_nc + enc_nc)

        # * Refinement module
        self.refine_block1 = RefinementBlock(enc_nc, use_dropout)
        self.refine_block2 = RefinementBlock(enc_nc, use_dropout)
        self.refine_block3 = RefinementBlock(enc_nc, use_dropout)
        self.refine_block4 = self.build_last_refine_block(64, out_channels)

        # * Output blocks
        self.output_block1 = self.build_output_block(512, out_channels)
        self.output_block2 = self.build_output_block(256, out_channels)
        self.output_block3 = self.build_output_block(128, out_channels)
        self.output_block4 = self.build_output_block(64, out_channels)

    def build_output_block(self, in_channels: int, out_channels: int):
        """Build output block of each U-Net decoder block
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns
        -------
        nn.Module
            The output block.
        """
        return nn.Sequential(
            [
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(in_channels),
                # * ------------------------------------
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
            ]
        )

    def build_pre_refine_block(
        self, in_channels: int, out_channels: int
    ) -> nn.Module:  # 3 -> 64
        """Build initial block
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels

        Returns
        -------
        nn.Module
            The pre-refinement block
        """
        return nn.Sequential(
            [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, stride=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1, stride=1
                ),
            ]
        )

    def build_last_refine_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build the last block of the refinement module

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns
        -------
        nn.Module
            The last block of the refinement module.
        """
        return nn.Sequential(
            [
                nn.Conv2d(
                    in_channels, in_channels, kernel_size=3, padding=1, bias=True
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels, in_channels, kernel_size=3, padding=1, bias=True
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels, in_channels, kernel_size=3, padding=1, bias=True
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=True
                ),
            ]
        )

    def forward(self, input_img, hist_enc1, hist_enc2):
        # * U-Net encode
        enc1, enc2, enc3, enc4, enc5 = self.UNetEnc(input_img)  # 1/3/256/256

        # * U-Net decode
        dec1, dec2, dec3, dec4, dec5 = self.UNetDec(
            enc1, enc2, enc3, enc4, enc5, hist_enc1, hist_enc2, input_img
        )

        # * Refinement
        pre_refine_out = self.pre_refine_block(input_img)
        tmp = self.refine_block1.forward(dec5 + pre_refine_out, hist_enc1, hist_enc2)
        tmp = self.refine_block2.forward(tmp + pre_refine_out, hist_enc1, hist_enc2)
        tmp = self.refine_block3.forward(tmp + pre_refine_out, hist_enc1, hist_enc2)
        out5 = self.refine_block4(tmp + pre_refine_out)

        out1 = self.output_block1(dec1)
        out2 = self.output_block2(dec2)
        out3 = self.output_block3(dec3)
        out4 = self.output_block4(dec4)

        return out1, out2, out3, out4, out5


def get_CTN(
    in_channels: int,
    out_channels: int,
    use_dropout=False,
    init_method="Normal",
) -> ColorTransferNetwork:
    """Construct and initialize weights of a CTN

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_dropout : bool, optional
        If use dropout, by default False.
    init_method : str, optional
        Weight initialization method, by default "Normal"

    Returns
    -------
    ColorTransferNetwork
        A CTN
    """
    model = ColorTransferNetwork(in_channels, out_channels, use_dropout).cuda()
    init_weights(model, type=init_method)

    return model
