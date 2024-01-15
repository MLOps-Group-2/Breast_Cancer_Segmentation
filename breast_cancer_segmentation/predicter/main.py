from fastapi import FastAPI
from hydra import compose, initialize
import logging
import torch
from breast_cancer_segmentation.models.UNETModel import UNETModel  # noqa


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
    # we return the top classes and the original image
    scores = unet_model(input)  # [1, 3, dim, dim]
    values, indices = torch.topk(scores, k=1, dim=1)
    return indices, input
