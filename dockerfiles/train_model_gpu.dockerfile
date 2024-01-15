# Base image
FROM nvcr.io/nvidia/pytorch:22.03-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
      apt clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN pip install -U setuptools

WORKDIR /

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY breast_cancer_segmentation/ breast_cancer_segmentation/
COPY config/hydra/ config/hydra/
# COPY data/ data/

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "breast_cancer_segmentation/trainer/train_model.py"]