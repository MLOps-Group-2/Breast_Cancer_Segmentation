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


def create_transforms():
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

    return train_imtrans, train_segtrans, val_imtrans, val_segtrans


def test_train_data():
    with initialize(version_base=None, config_path="./../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        print(os.getcwd())
        print(config)
        path = config.train_hyp["train_img_location"]
        print(path)
        train_images = sorted(glob(os.path.join(path, "*.png")))
        assert len(train_images) == 30760, "training data not of the correct len"
        assert isinstance(train_images, list), "Must load a list as train data"
        assert all(isinstance(x, str) for x in train_images), "all elements in train data list must be strings"


def test_train_mask_data():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        path = config.train_hyp["train_mask_location"]
        train_mask_images = sorted(glob(os.path.join(path, "*.png")))
        assert len(train_mask_images) == 30760, "training mask data not of correct len"
        assert isinstance(train_mask_images, list), "Must load a list as train mask"
        assert all(isinstance(x, str) for x in train_mask_images), "all elements in train mask list must be strings"


def test_val_data():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        path = config.train_hyp["validation_img_location"]
        val_images = sorted(glob(os.path.join(path, "*.png")))
        assert len(val_images) == 5429, "val images not of the correct len"
        assert isinstance(val_images, list), "Must load a list as val data"
        assert all(isinstance(x, str) for x in val_images), "all elements in val data list must be strings"


def test_val_mask_data():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        path = config.train_hyp["validation_mask_location"]
        val_mask_images = sorted(glob(os.path.join(path, "*.png")))
        assert len(val_mask_images) == 5429, "val mask images not of the correct len"
        assert isinstance(val_mask_images, list), "Must load a list as val mask"
        assert all(isinstance(x, str) for x in val_mask_images), "all elements in val mask list must be strings"


def test_test_data():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        path = config.train_hyp["test_location"]
        test_images = sorted(glob(os.path.join(path, "*.png")))
        assert len(test_images) == 4021, "test images not of the correct len"
        assert isinstance(test_images, list), "Must load a list as test data"
        assert all(isinstance(x, str) for x in test_images), "all elements in test data list must be strings"


def test_train_val_dataset():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        train_path = config.train_hyp["train_img_location"]
        train_mask_path = config.train_hyp["train_mask_location"]
        val_path = config.train_hyp["validation_img_location"]
        val_mask_path = config.train_hyp["validation_mask_location"]

        train_images = sorted(glob(os.path.join(train_path, "*.png")))
        train_mask_images = sorted(glob(os.path.join(train_mask_path, "*.png")))
        val_images = sorted(glob(os.path.join(val_path, "*.png")))
        val_mask_images = sorted(glob(os.path.join(val_mask_path, "*.png")))

        train_imtrans, train_segtrans, val_imtrans, val_segtrans = create_transforms()

        train_ds = ArrayDataset(train_images, train_imtrans, train_mask_images, train_segtrans)
        val_ds = ArrayDataset(val_images, val_imtrans, val_mask_images, val_segtrans)
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
