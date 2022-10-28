import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from .ColorTransferNetwork import get_CTN
from .HistogramEncodingNetwork import get_HEN
from .LearnableHistogram import LearnableHistogram

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../configs", config_name="DeepColorTransfer")
class DCT(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        """Initialize a Deep Color Transfer model.

        Parameters
        ----------
        cfg : DictConfig
            Config dictionary.
        """
        super(DCT, self).__init__()

        cfg = OmegaConf.to_yaml(cfg)

        self.l_bin = cfg.l_bin
        self.ab_bin = cfg.ab_bin
        self.img_type = cfg.img_type
        self.use_seg = cfg.use_seg
        self.pad = 30

        self.CTN = get_CTN(
            in_channels=3,
            out_channels=3,
            use_dropout=cfg.use_dropout,
            init_method=cfg.init_method,
        )
        self.HEN = get_HEN(
            in_channels=self.l_bin + 1, out_channels=64, init_method=cfg.init_method
        )

        self.histogram = LearnableHistogram(3)

    def _forward(
        self,
        input_img: torch.Tensor,  # (batch, 3, w1, h1)
        input_hist: torch.Tensor,  # (batch, l_bin+1, ab_bin, ab_bin)
        input_common_seg: torch.BoolTensor,  # (batch, num_seg_labels, w1, h1), 1 means label apears in both imgs, otherwise 0
        ref_img: torch.Tensor,  # (batch, 3, w2, h2)
        ref_hist: torch.Tensor,  # (batch, l_bin+1, ab_bin, ab_bin)
        ref_segwise_hist: torch.Tensor,  # (batch, num_seg_labels, l_bin+1, ab_bin, ab_bin)
    ):
        # * Get basic info
        batch_size, _, in_w, in_h = input_img.size()
        ref_w, ref_h = ref_img.size()[2:]

        # * ========================= HEN =========================

        # * ----------------- encoding histograms -----------------

        # * Encode histogram of input and reference image
        input_hist_enc = self.HEN(input_hist)
        ref_hist_enc = self.HEN(ref_hist)
        # * Tile the encoded histogram
        # (batch, 64, w1, h1)
        input_hist_enc_tile = input_hist_enc.repeat(1, 1, in_w, in_h)
        # (batch, 64, w2, h2)
        if not self.use_seg:
            ref_hist_enc_tile = ref_hist_enc.repeat(1, 1, ref_w, ref_h)

        # * ---------- segment-wise semantic replacement ----------

        else:
            # encoded segwise histogram(batch, nsl, 64, 1, 1)
            sw_hist_enc = self.HEN(
                ref_segwise_hist.view(-1, self.l_bin + 1, self.ab_bin, self.ab_bin)
            ).view(batch_size, -1, 64, 1, 1)
            # tiled encoded segwise histogram(batch, nsl, 64, w1, h1)
            sw_hist_enc_tile = sw_hist_enc.repeat(1, 1, 1, in_w, in_h)
            # * Replace common seg area value
            # (batch, 1, w1, h1)
            common_mask = input_common_seg.sum(dim=1).unsqueeze(1)
            replace_value = sw_hist_enc_tile[input_common_seg.unsqueeze(2)].sum(dim=1)

            # (batch, 64, w1, h1)
            ref_hist_enc_tile = ref_hist_enc.repeat(1, 1, in_w, in_h)
            ref_hist_enc_tile[common_mask] = replace_value

        # * -------------------- padding result -------------------

        input_HEN_out = self._rep_pad(input_hist_enc_tile)
        target_HEN_out = self._rep_pad(ref_hist_enc_tile)

        # * ========================= CTN =========================

        up_sample1, up_sample2, up_sample3, up_sample4, output_img = self.CTN(
            self._rep_pad(input_img), input_HEN_out, target_HEN_out
        )

        # * =================== return result =====================

        output_img = self._unpad(output_img)

        return input_img, up_sample1, up_sample2, up_sample3, up_sample4, output_img

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
        return F.pad(tensor, self.pad, "replicate")

    def _unpad(self, img: torch.Tensor) -> torch.Tensor:
        """De-padding image

        Parameters
        ----------
        img : torch.Tensor
            Target image

        Returns
        -------
        torch.Tensor
            De-padded image
        """
        w, h = img.size()[-2:]
        img = img[..., self.pad : w - 2 * self.pad, self.pad : h - 2 * self.pad]
        return img
