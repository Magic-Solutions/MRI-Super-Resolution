# Pulumi stack for Vertex training infra

This stack provisions persistent infrastructure for Vertex Custom Jobs:

- runtime service account,
- Artifact Registry Docker repository,
- bucket IAM grants for reading training data and writing artifacts,
- Secret Manager access for W&B API key,
- Cloud Build service-account IAM needed to push images to Artifact Registry and write logs,
- exported default worker profile (single A100).

## Quick start

```bash
cd infra/pulumi-gcp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pulumi stack init dev
cp Pulumi.dev.yaml.example Pulumi.dev.yaml
# edit values
pulumi up
pulumi stack output
```

Use stack outputs when calling `scripts/launch_vertex_training.py` and `scripts/submit_vertex_job.py`.

Note: this stack intentionally does not model each custom training execution as a Pulumi resource. Jobs are submitted dynamically by launcher scripts.

## W&B secret setup

Create the Secret Manager secret once (or add a version if it already exists):

```bash
printf '%s' "$WANDB_API_KEY" | gcloud secrets create wandb-api-key --data-file=-
# If secret already exists:
# printf '%s' "$WANDB_API_KEY" | gcloud secrets versions add wandb-api-key --data-file=-
```

Then set:

- `diamond-vertex-infra:wandbSecretId: wandb-api-key`

## Project-specific artifact bucket

Pulumi now always creates/manages the artifact bucket with a fixed name:

- `<project>-diamond-vertex-artifacts`
