import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .ColorTransferNetwork import get_CTN
from .HistogramEncodingNetwork import get_HEN
from .LearnableHistogram import LearnableHistogram, get_histogram2d

from typing import List


class DeepColorTransfer(nn.Module):
    def __init__(
        self,
        # * Model parameters
        l_bin: int,
        ab_bin: int,
        num_classes: int,
        use_seg: bool,
        hist_channels: int,
        init_method: str,
        encoder_name: str,
        CTN_enc_hidden_list: List[int],
        CTN_dec_hidden_list: List[int],
        HEN_hidden: int,
    ):
        """Initialize a Deep Color Transfer model.

        Parameters
        ----------
        """
        super(DeepColorTransfer, self).__init__()

        # * ---------------- Model hyper-parameters ---------------
        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.num_classes = num_classes
        self.use_seg = use_seg
        self.init_method = init_method
        # * value in original implementation is 30, change to 32 for easier calculation
        self.pad = 32
        self.hist_channels = hist_channels
        self.HEN_hidden = HEN_hidden

        # * -------------------- Define models --------------------
        self.HEN = get_HEN(
            in_channels=l_bin + 1,
            out_channels=hist_channels,
            hidden=HEN_hidden,
            init_method=init_method,
        )
        self.CTN = get_CTN(
            in_channels=3,
            out_channels=3,
            hist_channels=hist_channels,
            encoder_name=encoder_name,
            enc_hidden_list=CTN_enc_hidden_list,
            dec_hidden_list=CTN_dec_hidden_list,
            init_method=init_method,
        )
        self.histogram = LearnableHistogram(32)

    def forward(
        self,
        in_img: torch.Tensor,  # (batch, 3, h1, w1)
        in_hist: torch.Tensor,  # (batch, l_bin+1, ab_bin, ab_bin)
        in_common_seg: torch.BoolTensor,  # (batch, num_seg_labels, h1, w1), 1 means label apears in both imgs, otherwise 0
        ref_img: torch.Tensor,  # (batch, 3, h2, w2)
        ref_hist: torch.Tensor,  # (batch, l_bin+1, ab_bin, ab_bin)
        ref_segwise_hist: torch.Tensor,  # (batch, num_seg_labels, l_bin+1, ab_bin, ab_bin)
    ):
        # * Get basic info
        batch_size, _, in_w, in_h = in_img.size()
        ref_w, ref_h = ref_img.size()[2:]

        # * ========================= HEN =========================

        # * ----------------- encoding histograms -----------------

        # * Encode histogram of input and reference image
        input_hist_enc = self.HEN(in_hist)
        ref_hist_enc = self.HEN(ref_hist)
        # * Tile the encoded histogram
        # (batch, 64, h1, w1)
        input_hist_enc_tile = input_hist_enc.repeat(1, 1, in_w, in_h)
        # (batch, 64, h2, w2)
        if not self.use_seg:
            ref_hist_enc_tile = ref_hist_enc.repeat(1, 1, ref_w, ref_h)

        # * ---------- segment-wise semantic replacement ----------

        else:
            # encoded segwise histogram(batch, nsl, 64, 1, 1)
            sw_hist_enc = self.HEN(
                ref_segwise_hist.view(-1, self.l_bin + 1, self.ab_bin, self.ab_bin)
            ).view(batch_size, -1, self.hist_channels, 1, 1)
            # tiled encoded segwise histogram(batch, nsl, 64, h1, w1)
            sw_hist_enc_tile = sw_hist_enc.repeat(1, 1, 1, in_w, in_h)
            # * Replace common seg area value
            # (batch, 1, h1, w1)
            common_mask = in_common_seg.sum(dim=1).unsqueeze(1)
            # ( batch, 64, h1, w1)
            common_mask = common_mask.repeat(1, ref_hist_enc.size(1), 1, 1)
            in_common_seg = in_common_seg.unsqueeze(2).repeat(
                1, 1, sw_hist_enc_tile.size(2), 1, 1
            )

            replace_value = (sw_hist_enc_tile * in_common_seg).sum(dim=1)
            # (batch, 64, h1, w1)
            ref_hist_enc_tile = ref_hist_enc.repeat(1, 1, in_w, in_h)

            # print(sw_hist_enc_tile.size(), in_common_seg.size(), common_mask.size())
            # print(ref_hist_enc.size(), ref_hist_enc_tile.size(), replace_value.size())
            ref_hist_enc_tile = (
                ref_hist_enc_tile * (1 - common_mask) + common_mask * replace_value
            )

        # * -------------------- padding result -------------------
        # (batch, 64, h1+2*pad, w1+2*pad)
        in_HEN_out = self._rep_pad(input_hist_enc_tile)
        tar_HEN_out = self._rep_pad(ref_hist_enc_tile)

        # * ========================= CTN =========================

        CTN_out = self.CTN(self._rep_pad(in_img), in_HEN_out, tar_HEN_out)

        # * =================== return result =====================
        out_img = CTN_out[-1]
        out_img = self._unpad(out_img, self.pad)
        CTN_out[-1] = out_img

        return CTN_out

    def _rep_pad(self, tensor: torch.Tensor) -> torch.Tensor:
        """Replication padding

        Parameters
        ----------
        tensor : torch.Tensor
            Target tensor

        Returns
        -------
        torch.Tensor
            Padded tensor
        """
        return F.pad(tensor, (self.pad, self.pad, self.pad, self.pad), "replicate")

    def _unpad(self, img: torch.Tensor, pad: int) -> torch.Tensor:
        """De-padding image

        Parameters
        ----------
        img : torch.Tensor
            Target image
        pad : int
            Padding length

        Returns
        -------
        torch.Tensor
            De-padded image
        """
        w, h = img.size()[-2:]
        img = img[..., pad : w - pad, pad : h - pad]
        return img

    def calc_loss(
        self,
        label_img: torch.Tensor,
        decoder_out: List[torch.Tensor],
        lambda0: float,
        lambda1: float,
        lambda2: float,
    ) -> torch.Tensor:
        out_img = decoder_out[-1]
        # * image loss
        img_loss = (label_img - out_img) ** 2 * lambda0[..., None, None, None]
        img_loss = img_loss.mean()

        # * histogram loss
        out_hist = get_histogram2d(out_img, self.histogram)
        label_hist = get_histogram2d(label_img, self.histogram)
        hist_loss = F.mse_loss(out_hist, label_hist)

        # * multi-scale loss
        multi_loss = 0
        for i, dec_out in enumerate(decoder_out[:-1][::-1]):
            upsample = self._unpad(dec_out, self.pad // 2 ** (i + 1))
            label = T.Resize(upsample.size()[-2:])(label_img)
            multi_loss += F.mse_loss(upsample, label)
        multi_loss /= len(decoder_out) - 1

        return img_loss, lambda1 * hist_loss, lambda2 * multi_loss

    def soft_loss(
        self,
        teacher_out: List[torch.Tensor],
        decoder_out: List[torch.Tensor],
        lambda0: float,
        lambda1: float,
        lambda2: float,
    ) -> torch.Tensor:
        out_img = decoder_out[-1]
        # * image loss
        img_loss = (teacher_out[-1] - out_img) ** 2 * lambda0[..., None, None, None]
        img_loss = img_loss.mean()

        # * histogram loss
        out_hist = get_histogram2d(out_img, self.histogram)
        label_hist = get_histogram2d(teacher_out[-1], self.histogram)
        hist_loss = F.mse_loss(out_hist, label_hist)

        # * multi-scale loss
        multi_loss = 0
        for i, dec_out in enumerate(decoder_out[:-1]):
            multi_loss += F.mse_loss(dec_out, teacher_out[i])
        multi_loss /= len(decoder_out) - 1

        return img_loss, lambda1 * hist_loss, lambda2 * multi_loss
