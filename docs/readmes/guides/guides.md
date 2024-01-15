# General guides

## Vertex remote training

```bash
gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name=test-run \
    --config=config/vertex_jobs/config_training_cpu.yaml
```
