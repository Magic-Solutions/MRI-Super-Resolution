# CLAUDE.md

Code for *Comparative Analysis of 3D Convolutional and 2.5D Slice-Conditioned U-Net Architectures for MRI
Super-Resolution via Elucidated Diffusion Models* (Chiche, Gulati, Corcos, Rouge). The README has the results, setup
and commands; this file is the short version for coding agents.

## Layout

- `src/main.py`, `src/trainer.py`: 2.5D EDM upsampler (Hydra configs in `config/`).
- `src/train_3d.py`: 3D EDM upsampler.
- `src/train_swin2sr.py`: the EDSR and Swin2SR baselines (`--arch {edsr,swin2sr}`).
- `src/train_ressrdiff.py`, `src/ressrdiff_adapter.py`: the Res-SRDiff comparator, built from its authors' code,
  which is fetched at job start.
- `src/models/diffusion/`: EDM denoisers, slice conditioning and the shared sampler.
- `scripts/preprocess_fomo.py`: selection and preprocessing of the FOMO60K volumes; `splits/split.json` records the
  subject-level 500/6/70 split and how it was built.
- `scripts/evaluate_sr.py`: per-slice metrics for one method; `scripts/compare_sr.py`: per-subject statistics,
  bootstrap confidence intervals and Wilcoxon tests with Holm correction.
- `scripts/launch_vertex_training.py`, `scripts/run_training_job*.sh`: training on Vertex AI.
- `publication/`: the arXiv version of the paper (`preprint_arxiv.tex`) and its figures. The IEEE Access source is
  kept outside the repo.

## Conventions

- Every reported number comes from `scripts/evaluate_sr.py` and is aggregated per subject by `compare_sr.py`, never
  pooled over slices. Both EDM models use the shared deterministic Euler sampler (`--sampler shared`, 20 steps).
- Checkpoints are chosen on the validation subjects only; the test set never selects a checkpoint.
- The 2.5D model conditioned on both neighbouring slices needs
  `--override agent.upsampler.inner_model.num_next_slices_conditioning=1` at evaluation, as in training.
- Don't commit data, checkpoints, `.nii` files or credentials; `.gitignore` covers the data and checkpoint formats.
- Private working notes go in `CLAUDE.local.md`, which is gitignored.
