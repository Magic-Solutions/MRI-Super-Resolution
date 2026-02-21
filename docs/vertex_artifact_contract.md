# Vertex Run Artifact Contract

Each launched run writes to:

- `gs://<artifact-bucket>/<prefix>/runs/<run_name>/`

Expected structure:

- `recordings_manifest.txt`: exact selected recording list used for this run.
- `training_run/`: job-local training outputs copied from `/tmp/diamond_vertex/run`.
- `processed_data/`: generated `full_res/` and `low_res/` datasets used for training.

Recommended W&B metadata:

- `wandb.name = <run_name>`
- tags:
  - `vertex`
  - `a100`
  - `egocentric-rgbd`

Recommended summary metadata to log in W&B config:

- `gcs_output_uri`
- `recordings_manifest_uri`
- `depth_clip_min_mm`
- `depth_clip_max_mm`
