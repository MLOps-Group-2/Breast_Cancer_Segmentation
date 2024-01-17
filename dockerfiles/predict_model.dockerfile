# Base image
FROM python:3.11-slim

#RUN apt update && \
#    apt install --no-install-recommends -y build-essential gcc && \
#    apt clean && rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN set -e; \
    apt-get update -y && apt-get install -y \
    build-essential gcc wget curl \
    lsb-release; \
    gcsFuseRepo=gcsfuse-`lsb_release -c -s`; \
    echo "deb https://packages.cloud.google.com/apt $gcsFuseRepo main" | \
    tee /etc/apt/sources.list.d/gcsfuse.list; \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key add -; \
    apt-get update; \
    apt-get install -y gcsfuse \
    && apt-get clean

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY breast_cancer_segmentation/predicter breast_cancer_segmentation/predicter
COPY scripts/predicter_entrypoint.sh .

RUN chmod +x predicter_entrypoint.sh

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

CMD ["./predicter_entrypoint.sh"]
