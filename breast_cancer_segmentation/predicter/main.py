from fastapi import FastAPI, UploadFile, File
from PIL import Image
from http import HTTPStatus
from hydra import compose, initialize
import logging
import torch
import torchvision.transforms as transforms
from breast_cancer_segmentation.models.UNETModel import UNETModel  # noqa

log = logging.getLogger(__name__)

app = FastAPI()
with initialize(version_base=None, config_path="./config"):
    config = compose(config_name="config.yaml")
    # path to scripted neural net (with weights)
    model_path = config.environments.model_path
    unet_model = torch.jit.load(model_path)


@app.get("/health")
def read_health():
    return 200


@app.post("/predict-img/")
async def predict_img(data: UploadFile = File(...)):
    image = Image.open(data.file)
    transform = transforms.Compose(
        [
            # transforms.Resize((96, 96)),  # Resize to your desired input size (optional)
            transforms.ToTensor(),
        ]
    )
    # Apply the transform to the image
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)
    # print(image_tensor.shape)

    scores = unet_model(image_tensor)  # [1, 3, dim, dim]
    values, indices = torch.topk(scores, k=1, dim=1)
    # indices.squeeze(0)
    indices_2d = indices[0][0][:][:]
    print(indices_2d.shape)

    # somehow create a tensor with the image data
    # image_tensor = torch.frombuffer(image, dtype=float)
    # plt.show(image_tensor)
    # image.close()
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict")
def predict(input):
    # we return the top classes and the original image
    scores = unet_model(input)  # [1, 3, dim, dim]
    values, indices = torch.topk(scores, k=1, dim=1)
    return indices, input
    # return input
