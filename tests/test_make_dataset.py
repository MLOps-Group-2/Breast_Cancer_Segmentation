from glob import glob
import os
from hydra import initialize, compose
from monai.data import ArrayDataset
from monai.transforms import (
    Compose,
    LoadImage,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)


def test_train_data():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        path = config.train_hyp["train_data_path"]
        train_images = sorted(glob(os.path.join(path, "*.png")))
        assert len(train_images) == 30760, "training data not of the correct len"
        assert type(train_images) == list, "Must load a list as train data"
        assert all(isinstance(x, str) for x in train_images), "all elements in train data list must be strings"


def test_train_mask_data():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        path = config.train_hyp["train_mask_data_path"]
        train_mask_images = sorted(glob(os.path.join(path, "*.png")))
        assert len(train_mask_images) == 30760, "training mask data not of correct len"
        assert type(train_mask_images) == list, "Must load a list as train mask"
        assert all(isinstance(x, str) for x in train_mask_images), "all elements in train mask list must be strings"


def test_val_data():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        path = config.train_hyp["val_data_path"]
        val_images = sorted(glob(os.path.join(path, "*.png")))
        assert len(val_images) == 5429, "val images not of the correct len"
        assert type(val_images) == list, "Must load a list as val data"
        assert all(isinstance(x, str) for x in val_images), "all elements in val data list must be strings"


def test_val_mask_data():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        path = config.train_hyp["val_mask_data_path"]
        val_mask_images = sorted(glob(os.path.join(path, "*.png")))
        assert len(val_mask_images) == 5429, "val mask images not of the correct len"
        assert type(val_mask_images) == list, "Must load a list as val mask"
        assert all(isinstance(x, str) for x in val_mask_images), "all elements in val mask list must be strings"


def test_test_data():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        path = config.train_hyp["test_data_path"]
        test_images = sorted(glob(os.path.join(path, "*.png")))
        assert len(test_images) == 4021, "test images not of the correct len"
        assert type(test_images) == list, "Must load a list as test data"
        assert all(isinstance(x, str) for x in test_images), "all elements in test data list must be strings"


def test_train_val_dataset():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        train_path = config.train_hyp["train_data_path"]
        train_mask_path = config.train_hyp["train_mask_data_path"]
        val_path = config.train_hyp["val_data_path"]
        val_mask_path = config.train_hyp["val_mask_data_path"]

        train_images = sorted(glob(os.path.join(train_path, "*.png")))
        train_mask_images = sorted(glob(os.path.join(train_mask_path, "*.png")))
        val_images = sorted(glob(os.path.join(val_path, "*.png")))
        val_mask_images = sorted(glob(os.path.join(val_mask_path, "*.png")))

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
        print(val_ds[0][0].shape)
        print(val_ds[0][1].shape)
        assert train_ds[0][0].shape == (3, 96, 96) and train_ds[0][1].shape == (
            1,
            96,
            96,
        ), "Shape of training data is incorrect"
        assert val_ds[0][0].shape == (3, 224, 224) and val_ds[0][1].shape == (
            1,
            224,
            224,
        ), "Shape of validation data is incorrect"


"""
def test_train_loader():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")

        training_batch_size = 100
        training_num_workers = 8

        train_path = config.train_hyp['train_data_path']
        train_mask_path = config.train_hyp['train_mask_data_path']

        train_images = sorted(glob(os.path.join(train_path, "*.png")))
        train_mask_images = sorted(glob(os.path.join(train_mask_path, "*.png")))

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
        train_loader = DataLoader(
            train_ds,
            batch_size=training_batch_size,
            shuffle=True,
            num_workers=training_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        assert !train_loader, "DataLoader failed to be instantiated"
def test_val_loader():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
"""
