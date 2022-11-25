import torch
import torch.nn.functional as F
import torchvision.transforms as T

import pytorch_lightning as pl
from .ColorTransferNetwork import get_CTN
from .HistogramEncodingNetwork import get_HEN
from .LearnableHistogram import LearnableHistogram

from typing import List


class DCT(pl.LightningModule):
    def __init__(
        self,
        # * Model parameters
        l_bin: int,
        ab_bin: int,
        use_seg: bool,
        hist_channels: int,
        init_method: str,
        CTN_enc_hidden_list: List[int],
        CTN_dec_hidden_list: List[int],
        HEN_hidden: int,
        # * Optimization parameters
        loss_lambda1: float,
        loss_lambda2: float,
        learning_rate: float,
        beta1: float,
        beta2: float,
    ):
        """Initialize a Deep Color Transfer model.

        Parameters
        ----------
        cfg : str
            Config path.
        """
        super(DCT, self).__init__()

        # * ---------------- Model hyper-parameters ---------------
        self.l_bin = l_bin
        self.ab_bin = ab_bin
        self.use_seg = use_seg
        self.init_method = init_method
        # * value in original implementation is 30, change to 32 for easier calculation
        self.pad = 32
        self.hist_channels = hist_channels
        self.HEN_hidden = HEN_hidden

        # * ----------------- Training parameters -----------------
        self.loss_lambda1 = loss_lambda1
        self.loss_lambda2 = loss_lambda2
        self.learning_rate = learning_rate  # 5e-5
        self.beta1 = beta1  # 0.5
        self.beta2 = beta2  # 0.999

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
            enc_hidden_list=CTN_enc_hidden_list,
            dec_hidden_list=CTN_dec_hidden_list,
            init_method=init_method,
        )
        self.histogram = LearnableHistogram(3)

    def _forward(
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
            ).view(batch_size, -1, 64, 1, 1)
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

    def forward(self, x):
        output = self._forward(*x)
        return output[-1]

    # * ===================== training related ====================

    def calc_loss(
        self, label_img: torch.Tensor, decoder_out: List[torch.Tensor]
    ) -> torch.Tensor:
        out_img = decoder_out[-1]
        # * image loss
        img_loss = F.mse_loss(out_img, label_img)

        # * histogram loss
        hist_loss = F.mse_loss(self.histogram(out_img), self.histogram(label_img))

        # * multi-scale loss
        multi_loss = 0
        for i, dec_out in enumerate(decoder_out[:-1][::-1]):
            upsample = self._unpad(dec_out, self.pad // 2 ** (i + 1))
            label = T.Resize(upsample.size()[-2:])(label_img)
            multi_loss += F.mse_loss(upsample, label)
        multi_loss /= len(decoder_out) - 1

        # * total
        final_loss = (
            img_loss + self.loss_lambda1 * hist_loss + self.loss_lambda2 * multi_loss
        )

        return final_loss

    def training_step(self, batch, batch_idx):
        in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist = batch

        # * forward
        decoder_out = self._forward(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )

        # * calculate loss
        loss = self.calc_loss(ref_img, decoder_out)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist = batch

        # * forward
        decoder_out = self._forward(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )

        # * calculate loss
        loss = self.calc_loss(ref_img, decoder_out)

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2)
        )
        return optimizer

    # * predicting related

    def predict_step(self, batch, batch_idx):
        return self(batch)
