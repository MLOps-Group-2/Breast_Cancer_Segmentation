import pytorch_lightning as pl
from monai.data import decollate_batch

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose


class UNETModel(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, input):
        output = self.net(input)
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        if batch_idx % 5 == 0:
            print(f"training loss {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        val_images, val_labels = batch[0].to(self.device), batch[1].to(self.device)
        roi_size = (96, 96)
        sw_batch_size = 5
        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, self.net)
        val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]
        # compute metric for current iteration
        self.dice_metric(y_pred=val_outputs, y=val_labels)
        if batch_idx % 5 == 0:
            print(f"metric loss {self.dice_metric.aggregate().item()}")
        # aggregate the final mean dice result
        metric = self.dice_metric.aggregate().item()
        # reset the status for next validation round
        return metric
