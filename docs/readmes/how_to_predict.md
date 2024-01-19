# How to predict

Before starting to predict make sure to have succesfully completed all steps in [Getting Started](./getting_started.md).

## Locally using FastAPI

Open a terminal in this repository at root folder and do
```bash
make serve_api
```

Then go to http://127.0.0.1:8000/docs , click on predict-img and then the button `try out` . Choose an image to upload (eg. with 224x224 pixels) and click on `execute`. The result shows an image that has the predicted classes overlayed on the original image.

## Using Cloud Run API

To use the API hosted by Google Cloud Run one may use the following link: https://bcs-prediction-api-2jrai3q6wa-lz.a.run.app/docs

From the command line one would have to do:

```bash
curl -X 'POST' \
  'https://bcs-prediction-api-2jrai3q6wa-lz.a.run.app/predict-img/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'data=<LOCATION_OF_LOCAL_IMAGE>'
```
