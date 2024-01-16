import pytorch_lightning as pl
from monai.data import decollate_batch
import matplotlib.pyplot as plt
import torch

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose
from monai.visualize.utils import blend_images
import wandb


class UNETModel(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class, wandb_logging=False):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.wandb_logging = wandb_logging

    def forward(self, input):
        output = self.net(input)
        return output

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        val_images, val_labels = batch[0].to(self.device), batch[1].to(self.device)
        roi_size = (96, 96)
        sw_batch_size = 5
        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, self.net)
        val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]
        # compute metric for current iteration
        self.dice_metric(y_pred=val_outputs, y=val_labels)
        # create logging variables
        self.log("val_metric", self.dice_metric.aggregate().item())
        if batch_idx == 0 and self.wandb_logging:
            img_idx = 1
            values, indices = torch.topk(val_outputs[img_idx], k=1, dim=0)
            img = blend_images(val_images[img_idx], indices, transparent_background=True, cmap="YlGn")
            img_true = blend_images(val_images[img_idx], val_labels[img_idx], transparent_background=True, cmap="YlGn")
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img.permute(2, 1, 0).cpu())
            ax2.imshow(img_true.permute(2, 1, 0).cpu())
            ax1.axis("off")
            ax2.axis("off")
            ax1.title.set_text("prediction")
            ax2.title.set_text("ground truth")
            plt.tight_layout()
            self.logger.experiment.log({"prediction_image": wandb.Image(fig)})
        # aggregate the final mean dice result
        metric = self.dice_metric.aggregate().item()
        # reset the status for next validation round
        return metric
