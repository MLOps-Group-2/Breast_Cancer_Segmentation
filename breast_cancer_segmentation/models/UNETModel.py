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

    """
    def forward(self, input):
        output = self.net(input)
        return output
    """

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
            img_idx = [0, 1, 2, 3, 4]
            fig, axs = plt.subplots(5, 2)
            plt.rcParams["figure.dpi"] = 300
            for elem in img_idx:
                values, indices = torch.topk(val_outputs[elem], k=1, dim=0)
                img = blend_images(val_images[elem], indices, transparent_background=True, cmap="YlGn")
                img_true = blend_images(val_images[elem], val_labels[elem], transparent_background=True, cmap="YlGn")
                axs[elem, 0].imshow(img.permute(2, 1, 0).cpu())
                axs[elem, 1].imshow(img_true.permute(2, 1, 0).cpu())
                axs[elem, 0].axis("off")
                axs[elem, 1].axis("off")
                if elem == 0:
                    axs[elem, 0].title.set_text("prediction")
                    axs[elem, 1].title.set_text("ground truth")
            plt.tight_layout()
            self.logger.experiment.log({"prediction_image": wandb.Image(fig)})
            plt.close(fig)
        # aggregate the final mean dice result
        metric = self.dice_metric.aggregate().item()
        # reset the status for next validation round
        return metric
