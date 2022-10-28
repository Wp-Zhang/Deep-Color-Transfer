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
            # (b, c, h, w)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            # (b, c, h, w)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            # (b, c, h, w)
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            # (b, c, h/2, w/2)
        ]

        return nn.Sequential(*layers)

    def forward(self, input):
        # (b, c, h, w) -> (b, hidden, h/2, w/2)
        out1 = self.enc1.forward(input)
        # (b, hidden, h/2, w/2) -> (b, 2*hidden, h/4, w/4)
        out2 = self.enc2.forward(out1)
        # (b, 2*hidden, h/4, w/4) -> (b, 4*hidden, h/8, w/8)
        out3 = self.enc3.forward(out2)
        # (b, 4*hidden, h/8, w/8) -> (b, 8*hidden, h/16, w/16)
        out4 = self.enc4.forward(out3)
        # (b, 8*hidden, h/16, w/16) -> (b, 8*hidden, h/32, w/32)
        out5 = self.enc5.forward(out4)

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
                nn.Conv2d(channel_list[0], channel_list[1], kernel_size=3, padding=1),
                nn.InstanceNorm2d(channel_list[1]),
                # ------------------------------------------------
                nn.ReLU(True),
                nn.Conv2d(channel_list[1], channel_list[2], kernel_size=3, padding=1),
                nn.InstanceNorm2d(channel_list[2]),
                # ------------------------------------------------
                nn.ReLU(True),
                nn.Conv2d(channel_list[2], channel_list[3], kernel_size=3, padding=1),
                nn.InstanceNorm2d(channel_list[3]),
            ]
        )

    def forward(
        self,
        enc1: torch.Tensor,
        enc2: torch.Tensor,
        hist_enc1: torch.Tensor,
        hist_enc2: torch.Tensor,
        upsample_size: int,
    ) -> torch.Tensor:
        enc1 = F.upsample(enc1, size=upsample_size, mode="bilinear")
        enc2 = F.upsample(enc2, size=upsample_size, mode="bilinear")
        hist_enc1 = F.upsample(hist_enc1, size=upsample_size, mode="bilinear")
        hist_enc2 = F.upsample(hist_enc2, size=upsample_size, mode="bilinear")

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
        out1 = self.dec1(enc5, enc5, hist_enc1, hist_enc2, (enc5.size(2), enc5.size(3)))
        out2 = self.dec2(out1, enc4, hist_enc1, hist_enc2, (enc4.size(2), enc4.size(3)))
        out3 = self.dec3(out2, enc3, hist_enc1, hist_enc2, (enc3.size(2), enc3.size(3)))
        out4 = self.dec4(out3, enc2, hist_enc1, hist_enc2, (enc2.size(2), enc2.size(3)))
        out5 = self.dec5(out4, enc1, hist_enc1, hist_enc2, (enc1.size(2), enc1.size(3)))
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
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels * 3),
            nn.ReLU(True),
        ]

        if use_dropout:
            layers += [nn.Dropout(0.5)]

        layers += [
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1),
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hist_channels: int,
        enc_hidden: int = 64,
        use_dropout: bool = False,
    ):
        """Initialize a CTN
        Parameters
        ----------
        input_nc : int
            Number of input channels
        output_nc : int
            Number of output channels
        hist_channels : int
            Number of HEN output channels
        enc_hidden : int
            Number of hidden channels for U-Net encoder, by default 64
        use_dropout : bool, optional
            If use dropout, by default False
        """
        super(ColorTransferNetwork, self).__init__()

        # * Pre-refienment block
        self.pre_refine_block = self.build_pre_refine_block(in_channels, hist_channels)

        # * Encoder
        self.UNetEnc = UNetEncoder(in_channels, enc_hidden)

        # * Decoder
        self.UNetDec = UNetDecoder(enc_hidden * 8 * 2 + hist_channels * 2)

        # * Refinement module
        self.refine_block1 = RefinementBlock(hist_channels, use_dropout)
        self.refine_block2 = RefinementBlock(hist_channels, use_dropout)
        self.refine_block3 = RefinementBlock(hist_channels, use_dropout)
        self.refine_block4 = self.build_last_refine_block(hist_channels, out_channels)

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
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(in_channels),
                # * ------------------------------------
                nn.ReLU(True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ]
        )

    def build_pre_refine_block(self, in_channels: int, out_channels: int) -> nn.Module:
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
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ]
        )

    def forward(self, input_img, hist_enc1, hist_enc2):
        # * U-Net encode
        # (b, hidden, h/2, w/2)
        # (b, 2*hidden, h/4, w/4)
        # (b, 4*hidden, h/8, w/8)
        # (b, 8*hidden, h/16, w/16)
        # (b, 8*hidden, h/32, w/32)
        enc1, enc2, enc3, enc4, enc5 = self.UNetEnc(input_img)

        # * U-Net decode
        # (b, 512, h/16, w/16)
        # (b, 256, h/8, w/8)
        # (b, 128, h/4, w/4)
        # (b, 64, h/2, w/2)
        # (b, 64, h, w)
        dec1, dec2, dec3, dec4, dec5 = self.UNetDec(
            enc1, enc2, enc3, enc4, enc5, hist_enc1, hist_enc2, input_img
        )

        # * Refinement
        pre_refine_out = self.pre_refine_block(input_img)
        tmp = self.refine_block1.forward(dec5 + pre_refine_out, hist_enc1, hist_enc2)
        tmp = self.refine_block2.forward(tmp + pre_refine_out, hist_enc1, hist_enc2)
        tmp = self.refine_block3.forward(tmp + pre_refine_out, hist_enc1, hist_enc2)
        out5 = self.refine_block4(tmp + pre_refine_out)  # (b, 3, h, w)

        out1 = self.output_block1(dec1)  # (b, 3, h/16, w/16)
        out2 = self.output_block2(dec2)  # (b, 3, h/8, w/8)
        out3 = self.output_block3(dec3)  # (b, 3, h/4, w/4)
        out4 = self.output_block4(dec4)  # (b, 3, h/2, w/2)

        return out1, out2, out3, out4, out5


def get_CTN(
    in_channels: int,
    out_channels: int,
    hist_channels: int,
    enc_hidden: int = 64,
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
    hist_channels : int
        Number of HEN output channels
    enc_hidden : int
        Number of hidden channels for U-Net encoder, by default 64
    use_dropout : bool, optional
        If use dropout, by default False.
    init_method : str, optional
        Weight initialization method, by default "Normal"

    Returns
    -------
    ColorTransferNetwork
        A CTN
    """
    model = ColorTransferNetwork(
        in_channels, out_channels, hist_channels, enc_hidden, use_dropout
    ).cuda()
    init_weights(model, type=init_method)

    return model
