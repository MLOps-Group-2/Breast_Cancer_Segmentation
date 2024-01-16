import os
from hydra import compose, initialize

class Config:

    def __init__(self):
        with initialize(version_base=None, config_path="."):
            config = compose(config_name="config.yaml")
            self.model_path = os.getenv('MODEL_PATH', config.model_path)

