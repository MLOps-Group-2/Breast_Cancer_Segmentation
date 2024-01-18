from omegaconf import DictConfig
from glob import glob
import os
import hydra
import pandas as pd
import torch
import time
from monai.data import ArrayDataset
from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def get_class_distribution(labels: ArrayDataset) -> list:
    """Get images and compute class distribution
    @args
    dataset: arraydataset of tensors each representing an image

    @returns
    dataframe: dataframe of features extracted from dataset
    """
    count = [torch.bincount(torch.flatten(img.int()), minlength=3) for img in labels]
    return count


def get_grayscale_avg(dataset: ArrayDataset) -> list:
    """Get the images and extract grayscale average
    @args
    dataset: arraydataset of tensors each representing an image

    @returns
    list: list of grayscale avg value per image
    """
    dataset = torch.stack([torch.mean(image) for image in dataset])

    return dataset


def compute_features(grayscale, count: list = None) -> pd.DataFrame:
    """Get features and compute dataframe
    @args
    dataset: arraydataset of tensors each representing an image

    @returns
    dataframe: dataframe of features extracted from dataset
    """
    features = []
    columns = ["Grayscale"]
    features.append(grayscale)
    if count:
        features.append([data[0].item() for data in count])
        features.append([data[1].item() for data in count])
        columns.append("0")
        columns.append("1")
    dataframe = pd.DataFrame(features[0], columns=columns)
    return dataframe


def compute_data_drift_report(reference: pd.DataFrame, current: pd.DataFrame, location: str) -> None:
    """Generates and saves the data drift report using evidently

    @args
    reference: Dataframe containing reference data used in the past to train our model
    current: Dataframe containing current data that applies to our model

    @returns
    None
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    filename = "drift_check-" + time.strftime("%Y%m%d-%H%M") + ".html"
    report.save_html(location + filename)


@hydra.main(version_base=None, config_path="../../config/hydra", config_name="config_hydra.yaml")
def main(config: DictConfig):
    """Check data drifting robustness of our model

    @args
    config: hydra configuration for parameters

    @returns
    None
    """
    report_location = config.evidently.report_location
    train_images = sorted(glob(os.path.join(config.train_hyp.train_img_location, "*.png")))
    # train_masks = sorted(glob(os.path.join(config.train_hyp.train_mask_location, "*.png")))
    # val_masks = sorted(glob(os.path.join(config.train_hyp.validation_mask_location, "*.png")))
    test_images = sorted(glob(os.path.join(config.train_hyp.test_location, "*.png")))
    # define transforms for image and segmentation
    imtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
        ]
    )

    # Create datasets
    train_ds = ArrayDataset(train_images, imtrans)
    # train_masks_ds = ArrayDataset(train_masks, imtrans)
    test_ds = ArrayDataset(test_images, imtrans)
    # val_masks_ds = ArrayDataset(val_masks, imtrans)
    # counts = get_class_distribution(train_masks_ds)
    train_grayscale = get_grayscale_avg(train_ds[:10000])
    test_grayscale = get_grayscale_avg(test_ds)
    train_dataframe = compute_features(train_grayscale)
    test_dataframe = compute_features(test_grayscale)
    print("beginning report")
    compute_data_drift_report(train_dataframe, test_dataframe, report_location)


if __name__ == "__main__":
    main()
