import torch
import pytorch_lightning as pl
import torch.distributed as dist
from .DeepColorTransform import DeepColorTransfer
from ..data.transforms import LAB2RGB

from typing import List


class Model(pl.LightningModule):
    def __init__(
        self,
        # * Model parameters
        l_bin: int,
        ab_bin: int,
        num_classes: int,
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
        super(Model, self).__init__()
        self.save_hyperparameters()

        # * --------------------- Define model --------------------
        self.model = DeepColorTransfer(
            l_bin,
            ab_bin,
            num_classes,
            use_seg,
            hist_channels,
            init_method,
            CTN_enc_hidden_list,
            CTN_dec_hidden_list,
            HEN_hidden,
        )

        # * ----------------- Training parameters -----------------
        self.loss_lambda1 = loss_lambda1
        self.loss_lambda2 = loss_lambda2
        self.learning_rate = learning_rate  # 5e-5
        self.beta1 = beta1  # 0.5
        self.beta2 = beta2  # 0.999

    # * ===================== training related ====================

    def _post_process_img(self, img: torch.Tensor):
        img = (img * 0.5 + 0.5) * 255
        img = img.cpu().numpy()
        img = LAB2RGB(img.transpose(1, 2, 0))

        return img

    def training_step(self, batch, batch_idx):
        in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist = batch

        # * forward
        decoder_out = self.model(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )

        # * calculate loss
        loss = self.model.calc_loss(
            ref_img, decoder_out, self.loss_lambda1, self.loss_lambda2
        )

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist = batch

        # * forward
        decoder_out = self.model(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )

        if dataloader_idx == 0:
            # * Normal validation
            loss = self.model.calc_loss(
                ref_img, decoder_out, self.loss_lambda1, self.loss_lambda2
            )
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            return loss
        else:
            return in_img, ref_img, decoder_out[-1]

    def validation_epoch_end(self, outputs):

        if self.trainer.num_devices > 1:
            dist.barrier()
            demo_out = outputs[1]
            full_demo_out = [None for _ in self.trainer.device_ids]
            dist.all_gather_object(full_demo_out, demo_out)
        else:
            full_demo_out = outputs[1]

        if self.global_rank == 0:
            # * Visualization demo
            cnt = 0
            for device_out in full_demo_out:
                in_img, ref_img, decoder_out = device_out
                for i in range(len(in_img)):
                    in_img_demo = self._post_process_img(in_img[i])
                    ref_img_demo = self._post_process_img(ref_img[i])
                    out_demo = self._post_process_img(decoder_out[i])

                    self.logger.log_image(
                        key=f"Pair {cnt}",
                        images=[in_img_demo, ref_img_demo, out_demo],
                        caption=["Input", "Reference", "Output"],
                    )
                    cnt += 1

    def predict_step(self, batch, batch_idx):
        out = self.model(batch)[-1]

        res = []
        for img in out:
            img = self._post_process_img(img)
            res.append(img)
        return res

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2)
        )
        return optimizer
