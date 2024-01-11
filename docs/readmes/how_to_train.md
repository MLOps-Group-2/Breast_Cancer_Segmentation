# How to train

Before starting to train a model make sure to have succesfully completed all steps in [Getting Started](./getting_started.md).

# Local

To train a model locally on your machine execute the following command from the root of the project:
```bash
python ./breast_cancer_segmentation/trainer/train_model.py train_hyp=<config_file> model_hyp=<config_file>
```

Recommended config files for local training are [train_hyp_local](../../config/hydra/train_hyp/train_hyp_local.yaml) and [model_hyp_local](../../config/hydra/model_hyp/model_hyp_local.yaml). If you want to create your own config files place them in the same folder.

The results of the training and all logs will be stored in the [outputs](../../outputs)-folder.

# Local inside a Docker Container

# GCP