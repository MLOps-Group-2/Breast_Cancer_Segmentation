# Getting started

Before starting and setting up development environment we recommend having a look at the [system requirements](system_req.md) 
beforehand to make sure all requirements are installed.

Create conda environment from makefile
```bash
make create_environment
```

You should have the environment already activated and the terminal should have the following 
```bash
(breast_cancer_segmentation) $
```
Note if this not the case please make sure you activate the environment:
```bash
source activate breast_cancer_segmentation
```

Install the development dependancies to the environment
```bash
make dev_requirements
```