from fastapi import FastAPI
from hydra import compose, initialize
import logging
import torch


log = logging.getLogger(__name__)

app = FastAPI()
with initialize(version_bas=None, config_path="../../config/hydra"):
    config = compose(config_name="config_hydra.yaml")
    # path to scripted neural net (with weights)
    model_path = config.train_hyp.model_repo_location + config.predict_hyp.model_filename
    unet_model = torch.jit.load(model_path)


@app.get("/health")
def read_health():
    return 200


@app.post("/predict")
def read_item(input):
    # todo: we want to have top classes, not probs
    # todo: we want to get back original image as well
    return unet_model(input)
