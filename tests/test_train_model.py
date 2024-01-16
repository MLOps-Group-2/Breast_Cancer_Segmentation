from hydra import initialize, compose
from glob import glob
import os
import monai
from monai.data import ArrayDataset, DataLoader
from breast_cancer_segmentation.models.UNETModel import UNETModel
import torch
from monai.transforms import (
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
from torchtest import assert_vars_change


def create_dataloaders(config):
    train_img_location = config.unittest["train_img_location"]
    train_mask_location = config.unittest["train_mask_location"]
    validation_img_location = config.unittest["validation_img_location"]
    validation_mask_location = config.unittest["validation_mask_location"]

    training_batch_size = config.unittest["training_batch_size"]
    training_num_workers = config.unittest["training_num_workers"]
    validation_batch_size = config.unittest["validation_batch_size"]
    validation_num_workers = config.unittest["validation_num_workers"]

    train_images = sorted(glob(os.path.join(train_img_location, "*.png")))
    train_mask_images = sorted(glob(os.path.join(train_mask_location, "*.png")))
    val_images = sorted(glob(os.path.join(validation_img_location, "*.png")))
    val_mask_images = sorted(glob(os.path.join(validation_mask_location, "*.png")))

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

    train_ds = ArrayDataset(train_images, train_imtrans, train_mask_images, train_segtrans)
    val_ds = ArrayDataset(val_images, val_imtrans, val_mask_images, val_segtrans)

    train_loader = DataLoader(
        train_ds,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=training_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=validation_batch_size,
        num_workers=validation_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def define_model(config):
    # Define model hparams
    lr = config.train_hyp["learning_rate"]
    optimizer = torch.optim.AdamW

    # create UNet, DiceLoss and AdamW optimizer
    net = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    model = UNETModel(
        net=net,
        criterion=monai.losses.DiceCELoss(to_onehot_y=True, softmax=True),
        learning_rate=lr,
        optimizer_class=optimizer,
    )

    return model


def test_model_output():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        train_loader, val_loader = create_dataloaders(config)
        model = define_model(config)
        training_batch_size = config.unittest["training_batch_size"]

        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        outputs = model(images)
        # Only if softmax is applied in the model
        # We apply it in the loss, so no need for this assertion
        # assert torch.all(torch.isclose((torch.sum(outputs[0], dim=0)), torch.ones_like(torch.sum(outputs[0], dim=0)))), "The values for the 3 classes do not sum to one per every pixel"
        assert outputs.shape == (training_batch_size, 3, 96, 96), "Shape of model output is wrong"
        assert labels.shape == (training_batch_size, 1, 96, 96), "Shape of labels is wrong"


def test_parameters_update():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        train_loader, val_loader = create_dataloaders(config)
        model = define_model(config)

        dataiter = iter(train_loader)
        batch = next(dataiter)

        assert_vars_change(
            device=model.device,
            model=model,
            loss_fn=model.criterion,
            optim=model.optimizer_class(model.parameters(), lr=model.learning_rate),
            batch=batch,
        )


def check_loss_decrease():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        train_loader, val_loader = create_dataloaders(config)
        model = define_model(config)

        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        outputs = model(images)
        first_loss = model.criterion(outputs, labels)

        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        outputs = model(images)
        second_loss = model.criterion(outputs, labels)

        assert second_loss > first_loss, "Loss did not decrease"
