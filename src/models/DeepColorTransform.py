import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
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
        self.use_seg = cfg.use_seg
        # * value in original implementation is 30, change to 32 for easier calculation
        self.pad = 32
        self.hist_channels = 64

        self.HEN = get_HEN(
            in_channels=self.l_bin + 1,
            out_channels=self.hist_channels,
            init_method=cfg.init_method,
        )
        self.CTN = get_CTN(
            in_channels=3,
            out_channels=3,
            hist_channels=self.hist_channels,
            init_method=cfg.init_method,
        )

        self.histogram = LearnableHistogram(3)

        # * loss weights
        self.lambda1 = cfg.lambda1
        self.lambda2 = cfg.lambda2

    def _forward(
        self,
        in_img: torch.Tensor,  # (batch, 3, w1, h1)
        in_hist: torch.Tensor,  # (batch, l_bin+1, ab_bin, ab_bin)
        in_common_seg: torch.BoolTensor,  # (batch, num_seg_labels, w1, h1), 1 means label apears in both imgs, otherwise 0
        ref_img: torch.Tensor,  # (batch, 3, w2, h2)
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
            common_mask = in_common_seg.sum(dim=1).unsqueeze(1)
            replace_value = sw_hist_enc_tile[in_common_seg.unsqueeze(2)].sum(dim=1)

            # (batch, 64, w1, h1)
            ref_hist_enc_tile = ref_hist_enc.repeat(1, 1, in_w, in_h)
            ref_hist_enc_tile[common_mask] = replace_value

        # * -------------------- padding result -------------------
        # (batch, 64, w1+2*pad, h1+2*pad)
        in_HEN_out = self._rep_pad(input_hist_enc_tile)
        tar_HEN_out = self._rep_pad(ref_hist_enc_tile)

        # * ========================= CTN =========================

        upsample1, upsample2, upsample3, upsample4, out_img = self.CTN(
            self._rep_pad(in_img), in_HEN_out, tar_HEN_out
        )

        # * =================== return result =====================

        out_img = self._unpad(out_img)

        return in_img, upsample1, upsample2, upsample3, upsample4, out_img

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
        img = img[..., pad : w - 2 * pad, pad : h - 2 * pad]
        return img

    def loss(
        self,
        label_img: torch.Tensor,
        upsample1: torch.Tensor,
        upsample2: torch.Tensor,
        upsample3: torch.Tensor,
        upsample4: torch.Tensor,
        out_img: torch.Tensor,
    ) -> torch.Tensor:
        # * image loss
        img_loss = F.mse_loss(out_img, label_img)

        # * histogram loss
        hist_loss = F.mse_loss(self.histogram(out_img), self.histogram(label_img))

        # * multi-scale loss
        us1 = self._unpad(upsample1, 2)
        us2 = self._unpad(upsample2, 4)
        us3 = self._unpad(upsample3, 8)
        us4 = self._unpad(upsample4, 16)

        lbl1 = T.Resize((us1.size()[-2:]))(label_img)
        lbl2 = T.Resize((us2.size()[-2:]))(label_img)
        lbl3 = T.Resize((us3.size()[-2:]))(label_img)
        lbl4 = T.Resize((us4.size()[-2:]))(label_img)

        multi_loss = (
            F.mse_loss(us1, lbl1)
            + F.mse_loss(us2, lbl2)
            + F.mse_loss(us3, lbl3)
            + F.mse_loss(us4, lbl4)
        ) / 4

        # * total
        final_loss = img_loss + self.lambda1 * hist_loss + self.lambda2 * multi_loss

        return final_loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss
