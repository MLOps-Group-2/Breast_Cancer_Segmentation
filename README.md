# Breast Cancer Image Segmentation

1. [Overview](#overview)
2. [System Requirements](docs/readmes/system_req.md)
3. [Getting Started](docs/readmes/getting_started.md)
4. [How to Train](docs/readmes/how_to_train.md)
5. [How to Predict](docs/readmes/how_to_predict.md)

Project progress can be observed here:
[To-Do-List](docs/readmes/to_do_list.md)

## Overview

We perform segmentation of medical images to highlight presence of breast cancer.

This project is a part of a course in machine learning operations in the danish technical university. We will be working
on a segmentation of medical images to highlight presence of breast cancer. To accomplish this we use the
[Breast cancer semantic segmentation dataset](https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss/data)
provided on kaggle. The framework used to train the model is [MONAI](https://monai.io/) with the intent of a UNET architecture which
is popular for image segmentation in the medical domain.

![ML canvas](reports/ml-canvas-1.png "ML Canvas")

### Data

Our data used is the [Breast cancer semantic segmentation dataset](https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss/data)
specifically the **224x224** image sizes.

[APPLY IMAGE EXAMPLE]

The BCSS dataset, derived from TCGA, includes over 20,000 segmentation annotations of breast cancer tissue regions. Annotations are a collaborative effort of pathologists, residents, and medical students, using the Digital Slide Archive

#### Data version control (DVC)

For remote data version control we use a GCP blob as a data lake since we work with image data we need a file storage instead
of a traditional table storage.

### Modeling

The framework used to train the model is [MONAI](https://monai.io/), a PyTorch based framework for medical image analysis, that adds a level of abstraction to PyTorch. Instead of defining each layer of our own models, we can instead use entire networks (that are based on scientific papers from the medical research community) and only need to specifiy hyperparameters such as channel sizes, dimensions and loss functions. One such architecture family is [UNet](https://www.nature.com/articles/s41592-018-0261-2). We intend to use a UNET architecture as it is popular for image segmentation. We plan to use a [BasicUNet](https://docs.monai.io/en/stable/networks.html#basicunet) implementation first (based on CNN modules), and later potentially compare the performance to vision transformer based UNET like [UNetr](https://docs.monai.io/en/stable/networks.html#unetr) (this however is intended for 3D image data, so yet to be clarified).

### Containerization

The training procedure is containerized with docker utilizing the CUDA specific docker container for the option of GPU
accelerated training.

## Project structure

The project structure was initially created using the [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for the course machine learning operations
 course using the template [mlops_template](https://github.com/SkafteNicki/mlops_template).

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── project_name  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```
