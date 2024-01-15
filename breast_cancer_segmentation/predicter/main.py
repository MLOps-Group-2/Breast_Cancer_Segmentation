from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictionInputModel(BaseModel):
    value: str


@app.get("/health")
def read_health():
    return 200


@app.post("/predict")
def read_item(input: PredictionInputModel):
    return {"value": f"{input.value} world!"}
