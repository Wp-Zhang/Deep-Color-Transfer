import torch
import pytorch_lightning as pl
import torch.distributed as dist
from .DeepColorTransform import DeepColorTransfer
from ..data.transforms import LAB2RGB

from typing import List


class Model(pl.LightningModule):
    def __init__(
        self,
        teacher,
        soft_loss_weight,
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
        # * Optimization parameters
        loss_lambda0: float,
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
        self.teacher = teacher
        self.alpha = soft_loss_weight
        self.model = DeepColorTransfer(
            l_bin,
            ab_bin,
            num_classes,
            use_seg,
            hist_channels,
            init_method,
            encoder_name,
            CTN_enc_hidden_list,
            CTN_dec_hidden_list,
            HEN_hidden,
        )

        # * ----------------- Training parameters -----------------
        self.loss_lambda0 = loss_lambda0
        self.loss_lambda1 = loss_lambda1
        self.loss_lambda2 = loss_lambda2
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    # * ===================== training related ====================

    def _post_process_img(self, img: torch.Tensor):
        img = (img * 0.5 + 0.5) * 255
        img = img.cpu().numpy()
        img = LAB2RGB(img.transpose(1, 2, 0))

        return img

    def training_step(self, batch, batch_idx):
        (
            in_img,
            in_hist,
            in_common_seg,
            ref_img,
            ref_hist,
            ref_segwise_hist,
            is_identical,
        ) = batch

        # * forward
        decoder_out = self.model(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )
        teacher_out = self.teacher(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )

        # * calculate loss
        loss_weight0 = 1 - (1 - self.loss_lambda0) * is_identical.float()
        loss = self.model.calc_loss(
            ref_img,
            decoder_out,
            loss_weight0,
            self.loss_lambda1,
            self.loss_lambda2,
        )
        soft_loss = self.model.calc_loss(
            teacher_out,
            decoder_out,
            loss_weight0,
            self.loss_lambda1,
            self.loss_lambda2,
        )

        self.log("train_loss", sum(loss))
        self.log("train_img_loss", loss[0])
        self.log("train_hist_loss", loss[1])
        self.log("train_multi_loss", loss[2])

        self.log("train_tch_loss", sum(soft_loss))
        self.log("train_tch_img_loss", soft_loss[0])
        self.log("train_tch_hist_loss", soft_loss[1])
        self.log("train_tch_multi_loss", soft_loss[2])

        return sum(loss) * (1 - self.alpha) + self.alpha * sum(soft_loss)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        (
            in_img,
            in_hist,
            in_common_seg,
            ref_img,
            ref_hist,
            ref_segwise_hist,
            is_identical,
        ) = batch

        # * forward
        decoder_out = self.model(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )
        teacher_out = self.teacher(
            in_img, in_hist, in_common_seg, ref_img, ref_hist, ref_segwise_hist
        )

        if dataloader_idx == 0:
            # * Normal validation
            loss_weight0 = 1 - (1 - self.loss_lambda0) * is_identical.float()
            loss = self.model.calc_loss(
                ref_img,
                decoder_out,
                loss_weight0,
                self.loss_lambda1,
                self.loss_lambda2,
            )
            soft_loss = self.model.calc_loss(
                teacher_out,
                decoder_out,
                loss_weight0,
                self.loss_lambda1,
                self.loss_lambda2,
            )

            self.log("val_loss", sum(loss), prog_bar=True, sync_dist=True)
            self.log("val_img_loss", loss[0])
            self.log("val_hist_loss", loss[1])
            self.log("val_multi_loss", loss[2])

            self.log("val_tch_loss", sum(soft_loss), prog_bar=True, sync_dist=True)
            self.log("val_tch_img_loss", soft_loss[0])
            self.log("val_tch_hist_loss", soft_loss[1])
            self.log("val_tch_multi_loss", soft_loss[2])
            return sum(loss) * (1 - self.alpha) + self.alpha * sum(soft_loss)
        else:
            return in_img, ref_img, decoder_out[-1]

    def validation_epoch_end(self, outputs):

        if self.trainer.num_devices > 1:
            dist.barrier()
            demo_out = outputs[1]
            full_demo_out = [None for _ in self.trainer.device_ids]
            dist.all_gather_object(full_demo_out, demo_out)
        else:
            full_demo_out = [outputs[1]]

        if self.global_rank == 0:
            # * Visualization demo
            pair_names = {
                0: "High Relevance",
                1: "Weak Relevance",
                2: "No Relevance",
                3: "Hue Shift",
            }
            cnt = 0
            for device_out in full_demo_out:
                for batch in device_out:
                    in_img, ref_img, decoder_out = batch
                    for i in range(len(in_img)):
                        if cnt > 3:
                            return
                        in_img_demo = self._post_process_img(in_img[i])
                        ref_img_demo = self._post_process_img(ref_img[i])
                        out_demo = self._post_process_img(decoder_out[i])

                        self.logger.log_image(
                            key=pair_names[cnt],
                            images=[in_img_demo, ref_img_demo, out_demo],
                            caption=["Input", "Reference", "Output"],
                        )
                        cnt += 1

    def predict_step(self, batch, batch_idx):
        out = self.model(*batch)[-1]

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
