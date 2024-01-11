from gc import callbacks
import logging
import os
import sys
import tempfile
from glob import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar
#from pytorch_lightning import metrics
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from breast_cancer_segmentation.models.UNETModel import Our_UNETModel

import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
import hydra

from omegaconf import OmegaConf

def training_step():
    print("Hello World")

@hydra.main(version_base=None, config_path="./conf", config_name="config.yaml")
def main(config):
    """Initial training step"""
    # Ingest images from local file storage
    #print(OmegaConf.to_yaml(config))

    #Data params
    training_batch_size = 100
    validation_batch_size = 100
    testing_batch_size = 300
    training_num_workers = 8
    validation_num_workers = 4

    train_images = sorted(glob(os.path.join(config['resources']['dataset']['train_img_location'], "*.png")))
    train_segs = sorted(glob(os.path.join(config['resources']['dataset']['train_mask_location'], "*.png")))
    val_images = sorted(glob(os.path.join(config['resources']['dataset']['validation_img_location'], "*.png")))
    val_segs = sorted(glob(os.path.join(config['resources']['dataset']['validation_mask_location'], "*.png")))

    # define transforms for image and segmentation
    train_imtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            RandSpatialCrop((96, 96), random_size=False),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )
    train_segtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            RandSpatialCrop((96, 96), random_size=False),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )
    val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])

    # create a training data loader
    train_ds = ArrayDataset(train_images, train_imtrans, train_segs, train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=training_batch_size, shuffle=True, num_workers=training_num_workers, pin_memory=torch.cuda.is_available())
    # create a validation data loader
    val_ds = ArrayDataset(val_images, val_imtrans, val_segs, val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=validation_batch_size, num_workers=validation_num_workers, pin_memory=torch.cuda.is_available())


    #Define model hparams
    lr = 1e-2
    optimizer = torch.optim.AdamW

    # create UNet, DiceLoss and Adam optimizer
    net = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    model = Our_UNETModel(
    net=net,
    criterion=monai.losses.DiceCELoss(to_onehot_y = True, softmax=True),
    learning_rate=lr,
    optimizer_class=optimizer,
)

    #Define training params
    val_interval = 2
    max_epochs = 2
    limit_tb = 0.25 #Value from 0 to 1

    bar = ProgressBar()
    trainer = pl.Trainer(accelerator="auto", devices="auto", strategy="auto", limit_train_batches=limit_tb, max_epochs = max_epochs, log_every_n_steps=val_interval, callbacks=[bar])
    trainer.fit(model, train_loader, val_loader)
    #trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
