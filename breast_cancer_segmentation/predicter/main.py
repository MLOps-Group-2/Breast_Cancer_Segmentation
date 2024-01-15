from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from hydra import compose, initialize
import logging
import torch
import torchvision.transforms as transforms
from breast_cancer_segmentation.models.UNETModel import UNETModel  # noqa

log = logging.getLogger(__name__)

app = FastAPI()
with initialize(version_base=None, config_path="../../config/hydra"):
    config = compose(config_name="config_hydra.yaml")
    # path to scripted neural net (with weights)
    # model_path = config.train_hyp.model_repo_location + config.predict_hyp.model_filename
    # unet_model = None
    model_path = "./models/model-local.pt"
    unet_model = torch.jit.load(model_path)


@app.get("/health")
def read_health():
    return 200


@app.post("/predict-img/")
async def predict_img(data: UploadFile = File(...)):
    with open("image.jpg", "wb") as image:
        # content = await data.read()
        # image.write(content)
        pil_image = data.file.read()
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Adjust size according to your model's input size
                transforms.ToTensor(),
            ]
        )
        image_tensor = transform(pil_image)
        image_tensor = image_tensor.unsqueeze(0)

        scores = unet_model(input)  # [1, 3, dim, dim]
        values, indices = torch.topk(scores, k=1, dim=1)
        print(indices)

        # somehow create a tensor with the image data
        # image_tensor = torch.frombuffer(image, dtype=float)
        # plt.show(image_tensor)
        image.close()
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
