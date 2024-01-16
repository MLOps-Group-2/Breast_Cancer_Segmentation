# How to predict

Before starting to predict make sure to have succesfully completed all steps in [Getting Started](./getting_started.md).

## Locally using FastAPI

Start the local application.
```bash
uvicorn --reload --port 8000 ./breast_cancer_segmentation/predicter/main:app
```
Go to `http://localhost:8000/docs` and access the `predict-img` endpoint. Upload an image and receive your prediction.

## In the Cloud using Cloud Run
