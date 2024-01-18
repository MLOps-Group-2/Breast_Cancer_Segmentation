.PHONY: create_environment requirements dev_requirements clean data build_documentation serve_documentation train training_docker_build predict_docker_build predict_docker_run serve_api run_vertex_cpu run_remote_training_default

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = breast_cancer_segmentation
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python3.11
DOCKER_TRAINING_REPO = europe-docker.pkg.dev/igneous-thunder-410709/eu.gcr.io/bcs-trainer
DOCKER_PREDICT_REPO = europe-docker.pkg.dev/igneous-thunder-410709/eu.gcr.io/bcs-prediction-api

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) --no-default-packages -y
	source activate $(PROJECT_NAME)

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Install Developer Python Dependencies
dev_requirements: requirements
	$(PYTHON_INTERPRETER) -m pip install .["dev"]

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Run main training file
train:
	$(PYTHON_INTERPRETER) breast_cancer_segmentation/trainer/train_model.py

## Build training container
training_docker_build:
	docker build -t $(DOCKER_TRAINING_REPO):latest -f dockerfiles/train_model.dockerfile .

## Build training container
training_docker_run:
	docker run -d --rm --name $(PROJECT_NAME)_trainer $(DOCKER_TRAINING_REPO):latest -v ./data:data

## Build predicter container
predict_docker_build:
	docker build -t $(DOCKER_PREDICT_REPO):latest -f dockerfiles/predict_model.dockerfile .

## Run predicter container
predict_docker_run:
	docker run --rm --name $(PROJECT_NAME)_api -p 8000:80 $(DOCKER_PREDICT_REPO):latest

# predictor api
serve_api:
	uvicorn breast_cancer_segmentation.predicter.main:app

# Run vertex CPU
run_vertex_cpu:
	gcloud ai custom-jobs create --region=europe-north1 --display-name="BCSS-run" --config=./config/vertex_jobs/config_training_cpu.yaml

# Run default training remote
run_remote_training_default:
	gcloud compute ssh --zone "europe-west1-b" "training-instance" --project "igneous-thunder-410709" --command 'sh run_training.sh'

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Process raw data into processed data
data:
	python $(PROJECT_NAME)/data/make_dataset.py

#################################################################################
# Documentation RULES                                                           #
#################################################################################

## Build documentation
build_documentation: dev_requirements
	mkdocs build --config-file docs/mkdocs.yaml --site-dir build

## Serve documentation
serve_documentation: dev_requirements
	mkdocs serve --config-file docs/mkdocs.yaml

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
