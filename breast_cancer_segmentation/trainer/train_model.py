import os
import time
from glob import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


# from pytorch_lightning import metrics
from breast_cancer_segmentation.models.UNETModel import UNETModel

import monai
from monai.data import ArrayDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
import hydra
import logging
import omegaconf

log = logging.getLogger(__name__)


def training_step():
    print("Hello World")


@hydra.main(version_base=None, config_path="../../config/hydra", config_name="config_hydra.yaml")
def main(config):
    """Initial training step"""
    # Ingest images from local file storage
    # print(OmegaConf.to_yaml(config))

    # Data params
    training_batch_size = config.train_hyp.training_batch_size
    validation_batch_size = config.train_hyp.validation_batch_size
    testing_batch_size = 300  # noqa
    training_num_workers = config.train_hyp.training_num_workers
    validation_num_workers = config.train_hyp.validation_num_workers

    train_images = sorted(glob(os.path.join(config.train_hyp.train_img_location, "*.png")))
    train_segs = sorted(glob(os.path.join(config.train_hyp.train_mask_location, "*.png")))
    val_images = sorted(glob(os.path.join(config.train_hyp.validation_img_location, "*.png")))
    val_segs = sorted(glob(os.path.join(config.train_hyp.validation_mask_location, "*.png")))

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
    train_loader = DataLoader(
        train_ds,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=training_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = ArrayDataset(val_images, val_imtrans, val_segs, val_segtrans)
    val_loader = DataLoader(
        val_ds,
        batch_size=validation_batch_size,
        num_workers=validation_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Define model hparams
    lr = config.train_hyp.learning_rate
    if config.train_hyp.optimizer == "AdamW":
        optimizer = torch.optim.AdamW
    elif config.train_hyp.optimizer == "Adam":
        optimizer = torch.optim.Adam
    else:
        log.error("No valid optimizer name in configuration file")

    # create UNet, DiceLoss and Adam optimizer
    if config.model_hyp.model == "UNet":
        net = monai.networks.nets.UNet(
            spatial_dims=config.model_hyp.spatial_dims,
            in_channels=config.model_hyp.in_channels,
            out_channels=config.model_hyp.out_channels,
            channels=config.model_hyp.channels,
            strides=config.model_hyp.strides,
            num_res_units=config.model_hyp.num_res_units,
            dropout=config.model_hyp.dropout,
            act=config.model_hyp.activation,
            kernel_size=config.model_hyp.kernel_size,
            up_kernel_size=config.model_hyp.up_kernel_size,
        )
    elif config.model_hyp.model == "AttentionUnet":
        net = monai.networks.nets.AttentionUnet(
            spatial_dims=config.model_hyp.spatial_dims,
            in_channels=config.model_hyp.in_channels,
            out_channels=config.model_hyp.out_channels,
            channels=config.model_hyp.channels,
            strides=config.model_hyp.strides,
            dropout=config.model_hyp.dropout,
            kernel_size=config.model_hyp.kernel_size,
            up_kernel_size=config.model_hyp.up_kernel_size,
        )
    else:
        log.error("No valid model name in configuration file")

    model = UNETModel(
        net=net,
        criterion=monai.losses.DiceCELoss(to_onehot_y=True, softmax=True),
        learning_rate=lr,
        optimizer_class=optimizer,
        wandb_logging=config.train_hyp.wandb_enabled,
    )

    # Define training params
    val_interval = 2  # noqa
    max_epochs = config.train_hyp.max_epochs
    limit_tb = config.train_hyp.limit_train_batches  # Value from 0 to 1

    if config.train_hyp.wandb_enabled:
        wandb_logger = WandbLogger(
            project="dtu_mlops_group2",
            save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            config=omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            anonymous="allow",
        )
    else:
        wandb_logger = False

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        limit_train_batches=limit_tb,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        logger=wandb_logger,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)

    filename = "/model-" + time.strftime("%Y%m%d-%H%M") + ".pt"

    # Save the model in TorchScript format
    script = model.to_torchscript()

    torch.jit.save(script, config.train_hyp.model_repo_location.strip() + filename)


if __name__ == "__main__":
    main()
