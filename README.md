# Breast Cancer Image Segmentation

## Software Requirements

- python >= 3.10
- pip >=
- docker
- gcp command line interface
- conda >=
- 

## Overview (PROPOSAL)

We perform segmentation of medical images to highlight presence of breast cancer.

This project is a part of a course in machine learning operations in the danish technical university. We will be working
on a segmentation of medical images to highlight presence of breast cancer. To accomplish this we use the 
[Breast cancer semantic segmentation dataset](https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss/data) 
provided on kaggle. The framework used to train the model is [MONAI](https://monai.io/) with the intent of a UNET architecture which 
is popular for image segmentation

![ML canvas](reports/ml-canvas-1.png "ML Canvas")

### Data

Our data used is the [Breast cancer semantic segmentation dataset](https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss/data) 
specifically the **224x224** image sizes. 

[APPLY IMAGE EXAMPLE]

[APPLY DATASET DESCRIPTION]

#### Data version control (DVC)



### Modeling

The framework used to train the model is [MONAI](https://monai.io/) with the intent of a UNET architecture which 
is popular for image segmentation and to compare vision transformer based UNET with CNN based UNET.

### Containerization

The training procedure is containerized with docker utilizing the CUDA specific docker container for the option of GPU 
accellirated training.

### 

# Project proposal
- Overwiew:
    - implement a semantic segmentation network to segment images from a breast cancer data set
- Modeling: 
    - we intend to use a UNET architecture which is popular for medical image segmentation
    - we plan to compare (vision) transformer based UNET with CNN based UNET
  
- Code Organization
    - cookie clutter template

- Containerization:
    - in google cloud: run inside a Docker container (possibly a training container and a prediction container --> makes productive use slimer, no trianing dependencies in deployed model container)
    - 

- Configuration handling
    - ?

- Data handling:
    - we DVC for data versioning
    - we will use blobs/buckets on google cloud
   	

- Training:
    - use google cloud (CPU or if GPU then K80/something cheap)
    - wandb/other tools compatible/available in google cloud

- Deployment:
    - use something like FastAPI to deploy model/be able to access it
    
- Monitoring:
    - during production

--------------------------------------------


Next steps:
- TODO

--------------------------------------------

Overall goal of the project
What framework are you going to use and you do you intend to include the framework into your project?
What data are you going to run on (initially, may change)
What models do you expect to use



## Project structure

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

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
