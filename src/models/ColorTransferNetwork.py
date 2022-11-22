from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from ..util import init_weights

# * ============================== Modules ==============================


class UNetEncoder(nn.Module):
    """U-Net Encoder"""

    def __init__(self, in_channels: int, hidden_list: List[int]):
        """Initialize a U-Net encoder

        Parameters
        ----------
        in_channels : int
            Number of input channels
        hidden_list : List[int]
            List of hidden channels in each layer
        """
        super(UNetEncoder, self).__init__()

        self.enc = []
        for i, hidden in enumerate(hidden_list):
            if i == 0:
                self.enc.append(self.build_block(in_channels, hidden, is_first=True))
            else:
                last_hidden = hidden_list[i - 1]
                self.enc.append(self.build_block(last_hidden, hidden, False))

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
        out = []
        tmp_out = input
        for i, layer in enumerate(self.enc):
            # (b, hidden_(i-1), h/2^i, w/2^i) -> (b, hidden_i, h/2^(i+1), w/2^(i+1))
            tmp_out = layer(tmp_out)
            out.append(tmp_out)

        return out


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

    def __init__(self, enc_hidden_list: List[int], dec_hidden_list: List[int], hist_nc):
        super(UNetDecoder, self).__init__()

        self.dec = []
        for i, enc_hidden in enumerate(enc_hidden_list[::-1]):
            dec_hidden = dec_hidden_list[i]
            block = UNetDecoderBlock(
                [enc_hidden * 2 + hist_nc * 2, dec_hidden, dec_hidden, dec_hidden]
            )  # TODO a little different with the original version
            self.dec.append(block)

    def forward(self, enc_list, hist_enc1, hist_enc2, input_img):
        out_list = []
        enc_list = enc_list[::-1]

        out = enc_list[0]
        for i, decoder in enumerate(self.dec):
            dec = enc_list[i]
            out = decoder(out, dec, hist_enc1, hist_enc2, dec.size()[2:])
            out_list.append(out)

        out_list[-1] = F.upsample(
            out_list[-1], size=input_img.size()[2:], mode="bilinear"
        )

        return out_list


class RefinementBlock(nn.Module):
    """Refinement module block"""

    def __init__(self, hist_channels: int, use_dropout=False):
        """Initialize a refinement block

        Parameters
        ----------
        hist_channels : int
            Number of HEN output channels
        use_dropout : bool, optional
            If use dropout, by default False
        """
        super(RefinementBlock, self).__init__()

        layers = [
            nn.Conv2d(hist_channels * 3, hist_channels * 3, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hist_channels * 3),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        layers += [
            nn.Conv2d(hist_channels * 3, hist_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hist_channels),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x, hist_enc1, hist_enc2):
        hist_enc1 = F.upsample(hist_enc1, size=x.size()[2:], mode="bilinear")
        hist_enc2 = F.upsample(hist_enc2, size=x.size()[2:], mode="bilinear")

        x_cat = torch.cat((x, hist_enc1, hist_enc2), 1)
        out = x + self.block(x_cat)  # 3*hist_channels -> hist_channels
        return out


# * ======================= ColorTransferNetwork =======================


class ColorTransferNetwork(nn.Module):
    """Color Transfer Network"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hist_channels: int,
        enc_hidden_list: List[int] = [64, 128, 256, 512, 512],
        dec_hidden_list: List[int] = [512, 256, 128, 64, 64],
        use_dropout: bool = False,
    ):
        """Initialize a CTN
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
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
        self.UNetEnc = UNetEncoder(in_channels, enc_hidden_list)

        # * Decoder
        self.UNetDec = UNetDecoder(enc_hidden_list, dec_hidden_list, hist_channels)

        # * Refinement module
        self.refine_block1 = RefinementBlock(hist_channels, use_dropout)
        self.refine_block2 = RefinementBlock(hist_channels, use_dropout)
        self.refine_block3 = RefinementBlock(hist_channels, use_dropout)
        self.refine_block4 = self.build_last_refine_block(hist_channels, out_channels)

        # * Output blocks
        self.output_blocks = []
        for hidden in dec_hidden_list[:-1]:
            block = self.build_output_block(hidden, out_channels)
            self.output_blocks.append(block)

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
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            # * ------------------------------------
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, input_img, hist_enc1, hist_enc2):
        # * U-Net encode
        # (b, hidden, h/2, w/2)
        # (b, 2*hidden, h/4, w/4)
        # (b, 4*hidden, h/8, w/8)
        # (b, 8*hidden, h/16, w/16)
        # (b, 8*hidden, h/32, w/32)
        enc_list = self.UNetEnc(input_img)

        # * U-Net decode
        # (b, 512, h/16, w/16)
        # (b, 256, h/8, w/8)
        # (b, 128, h/4, w/4)
        # (b, 64, h/2, w/2)
        # (b, 64, h, w)
        dec_list = self.UNetDec(enc_list, hist_enc1, hist_enc2, input_img)

        # * Refinement
        last = dec_list[-1]
        pre_refine_out = self.pre_refine_block(input_img)
        tmp = self.refine_block1.forward(last + pre_refine_out, hist_enc1, hist_enc2)
        tmp = self.refine_block2.forward(tmp + pre_refine_out, hist_enc1, hist_enc2)
        tmp = self.refine_block3.forward(tmp + pre_refine_out, hist_enc1, hist_enc2)
        last = self.refine_block4(tmp + pre_refine_out)  # (b, 3, h, w)
        dec_list[-1] = last

        for i, block in enumerate(self.output_blocks):
            dec_list[i] = block(dec_list[i])

        return dec_list


def get_CTN(
    in_channels: int,
    out_channels: int,
    hist_channels: int,
    enc_hidden_list: List[int] = [64, 128, 256, 512, 512],
    dec_hidden_list: List[int] = [512, 256, 128, 64, 64],
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
    enc_hidden_list : List[int]
        Number of hidden channels for U-Net encoder, by default [64, 128, 256, 512, 512]
    dec_hidden_list : List[int]
        Number of hidden channels for U-Net decoder, by default [512, 256, 128, 64, 64]
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
        in_channels,
        out_channels,
        hist_channels,
        enc_hidden_list,
        dec_hidden_list,
        use_dropout,
    )
    init_weights(model, type=init_method)

    return model
