from fastapi import FastAPI, UploadFile, File, Response
from PIL import Image
from io import BytesIO


import logging
import torch
import torchvision.transforms as transforms
#from breast_cancer_segmentation.models.UNETModel import UNETModel  # noqa
from monai.visualize.utils import blend_images
from .config.Config import Config
from breast_cancer_segmentation.utils.gcp import download_blob

log = logging.getLogger(__name__)

app = FastAPI()

config = Config()

# Load model
if config.storage_mode == "gcp":
    download_blob(config.model_repository, config.model_path, './models/model.pt')
unet_model = torch.jit.load(config.model_path)

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
    image_tensor_no_batch = transform(image)

    image_tensor = image_tensor_no_batch.unsqueeze(0)
    # print(image_tensor.shape)

    scores = unet_model(image_tensor)  # [1, 3, dim, dim]
    values, indices = torch.topk(scores, k=1, dim=1)

    # indices.squeeze(0)
    indices_2d = indices[0][:][:]
    print(f"Labels Shape: {indices_2d.shape}")
    print(f"predicted classes: {set(indices_2d.numpy().flatten())}")
    print(f"Input Image Shape: {image_tensor_no_batch.shape}")

    blended_image = blend_images(image_tensor_no_batch, indices_2d, transparent_background=True, cmap="YlGn")  # noqa

    pil_blended = transforms.ToPILImage()(blended_image)

    with BytesIO() as output_bytes:
        pil_blended.save(output_bytes, format="JPEG")
        blended_bytes = output_bytes.getvalue()

    return Response(content=blended_bytes, media_type="image/png")

    # todo: display/send back blended image
    # response = {
    #    "input": data,
    #    "message": HTTPStatus.OK.phrase,
    #    "status-code": HTTPStatus.OK,
    # }
    #
    # return response


@app.post("/predict")
def predict(input):
    # we return the top classes and the original image
    scores = unet_model(input)  # [1, 3, dim, dim]
    values, indices = torch.topk(scores, k=1, dim=1)
    return indices, input
