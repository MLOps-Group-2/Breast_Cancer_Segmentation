# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY breast_cancer_segmentation/ breast_cancer_segmentation/
COPY config/hydra/ config/hydra/
COPY reports/ reports/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "breast_cancer_segmentation/visualizations/data_drifting.py"]
