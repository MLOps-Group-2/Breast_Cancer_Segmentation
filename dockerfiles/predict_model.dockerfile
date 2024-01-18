# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY breast_cancer_segmentation/predicter breast_cancer_segmentation/predicter

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

CMD ["uvicorn", "breast_cancer_segmentation.predicter.main:app", "--host", "0.0.0.0", "--port", "80"]
