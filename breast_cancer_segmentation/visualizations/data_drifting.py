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
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def compute_features(dataset: ArrayDataset) -> pd.DataFrame:
    """Get the images and extract meaningful features to check data drifting
    @args
    dataset: arraydataset of tensors each representing an image

    @returns
    dataframe: dataframe of features extracted from dataset
    """
    dataset = torch.stack([torch.mean(image) for image in dataset])
    dataframe = pd.DataFrame(dataset, columns=["Grayscale"])
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
    report_location = config.train_hyp.report_location
    train_images = sorted(glob(os.path.join(config.train_hyp.train_img_location, "*.png")))
    test_images = sorted(glob(os.path.join(config.train_hyp.test_location, "*.png")))
    # define transforms for image and segmentation
    imtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            RandSpatialCrop((96, 96), random_size=False),
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )

    # Create datasets
    train_ds = ArrayDataset(train_images, imtrans)
    test_ds = ArrayDataset(test_images, imtrans)

    pd_train_data = compute_features(train_ds[:4000])
    pd_test_data = compute_features(test_ds)
    compute_data_drift_report(pd_train_data, pd_test_data, report_location)


if __name__ == "__main__":
    main()
