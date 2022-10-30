import torch
import torch.nn.functional as F
import torchvision.transforms as T

import pytorch_lightning as pl
from .ColorTransferNetwork import get_CTN
from .HistogramEncodingNetwork import get_HEN
from .LearnableHistogram import LearnableHistogram

from box import Box


class DCT(pl.LightningModule):
    def __init__(self, cfg_path: str):
        """Initialize a Deep Color Transfer model.

        Parameters
        ----------
        cfg : str
            Config path.
        """
        super(DCT, self).__init__()

        cfg = Box.from_yaml(open(cfg_path, "r").read())

        # * ---------------- Model hyper-parameters ---------------
        self.l_bin = cfg.l_bin
        self.ab_bin = cfg.ab_bin
        self.use_seg = cfg.use_seg
        self.init_method = cfg.init_method
        # * value in original implementation is 30, change to 32 for easier calculation
        self.pad = 32
        self.hist_channels = 64

        # * ----------------- Training parameters -----------------
        self.lambda1 = cfg.lambda1
        self.lambda2 = cfg.lambda2
        self.learning_rate = cfg.learning_rate  # 5e-5
        self.beta1 = cfg.beta1  # 0.5
        self.beta2 = cfg.beta2  # 0.999

        # * -------------------- Define models --------------------
        self.HEN = get_HEN(
            in_channels=self.l_bin + 1,
            out_channels=self.hist_channels,
            init_method=self.init_method,
        )
        self.CTN = get_CTN(
            in_channels=3,
            out_channels=3,
            hist_channels=self.hist_channels,
            init_method=self.init_method,
        )
        self.histogram = LearnableHistogram(3)

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
            # ( batch, 64, w1, h1)
            common_mask = common_mask.repeat(1, ref_hist_enc.size(1), 1, 1)
            in_common_seg = in_common_seg.unsqueeze(2).repeat(
                1, 1, sw_hist_enc_tile.size(2), 1, 1
            )

            replace_value = (sw_hist_enc_tile * in_common_seg).sum(dim=1)
            # (batch, 64, w1, h1)
            ref_hist_enc_tile = ref_hist_enc.repeat(1, 1, in_w, in_h)

            # print(sw_hist_enc_tile.size(), in_common_seg.size(), common_mask.size())
            # print(ref_hist_enc.size(), ref_hist_enc_tile.size(), replace_value.size())
            ref_hist_enc_tile = (
                ref_hist_enc_tile * (1 - common_mask) + common_mask * replace_value
            )

        # * -------------------- padding result -------------------
        # (batch, 64, w1+2*pad, h1+2*pad)
        in_HEN_out = self._rep_pad(input_hist_enc_tile)
        tar_HEN_out = self._rep_pad(ref_hist_enc_tile)

        # * ========================= CTN =========================

        upsample1, upsample2, upsample3, upsample4, out_img = self.CTN(
            self._rep_pad(in_img), in_HEN_out, tar_HEN_out
        )

        # * =================== return result =====================

        out_img = self._unpad(out_img, self.pad)
        return upsample1, upsample2, upsample3, upsample4, out_img

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

    def forward(self, x):
        in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist = x
        _, _, _, _, out = self._forward(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )
        return out

    # * ===================== training related ====================

    def calc_loss(
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
        in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist = batch

        # * forward
        decoder_out = self._forward(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )

        # * calculate loss
        loss = self.calc_loss(ref_img, *decoder_out)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist = batch

        # * forward
        decoder_out = self._forward(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )

        # * calculate loss
        loss = self.calc_loss(ref_img, *decoder_out)

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2)
        )
        return optimizer
