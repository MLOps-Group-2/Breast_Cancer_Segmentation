import requests
from hydra import initialize, compose
from glob import glob
import os
from PIL import Image
from io import BytesIO


def test_prediction():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        url = config.unittest.api_url
        train_img_location = config.unittest.train_img_location

        train_images = sorted(glob(os.path.join(train_img_location, "*.png")))

        files = {
            "data": open(train_images[0], "rb"),
            "Content-Type": "image/jpeg",
        }

        response = requests.post(url, files=files)
        image_bytes = BytesIO(response.content)  # noqa
        pil_image = Image.open(image_bytes)  # noqa
        assert response.status_code == 200, f"Response error {response.status_code}"


def test_response():
    with initialize(version_base=None, config_path="../config/hydra/"):
        config = compose(config_name="config_hydra.yaml")
        url = config.unittest.api_url
        train_img_location = config.unittest.train_img_location

        train_images = sorted(glob(os.path.join(train_img_location, "*.png")))

        files = {
            "data": open(train_images[0], "rb"),
            "Content-Type": "image/jpeg",
        }

        response = requests.post(url, files=files)
        image_bytes = BytesIO(response.content)  # noqa
        pil_image = Image.open(image_bytes)  # noqa
        assert pil_image.size == Image.open(train_images[0]).size
        assert response.status_code == 200, f"Response error {response.status_code}"
