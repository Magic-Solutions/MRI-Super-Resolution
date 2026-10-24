# CLAUDE.md — MRI Super-Resolution: resubmission to IEEE Access

## Goal

This repo holds the code for the manuscript *"Comparative Analysis of 3D Convolutional and 2.5D Slice-Conditioned U-Net Architectures for MRI Super-Resolution via Elucidated Diffusion Models"* (IEEE Access, Manuscript ID Access-2026-10274). The paper was **not accepted** (decision 7 May 2026). The editor encouraged a resubmission.

**IEEE Access allows only ONE resubmission.** If the revised paper does not address *every* reviewer concern, it is rejected for good. So for each comment we must either (a) run the experiment, (b) change the text, or (c) give a reasoned rebuttal in the response letter.

The work has two parts:
1. **Triage** every reviewer comment: decide whether to address it with experiments, with text changes, or with a rebuttal. See the triage table below.
2. **Run** the selected experiments, then update the manuscript and write the response-to-reviewers document. That document needs, for each comment: the concern, our response, and the action taken.

Sources:
- Reviewer comments: `publication/reviewer-comments.html` (5 reviewers; R5 also attached `Access-2026-10274_Proof_hi_flaws.pdf`, which is not in the repo yet and needs to be obtained).
- Manuscript: `publication/manuscript_ieee_access/preprint_arxiv.tex`.

## Infrastructure

- GCP project: **`omgrab-dev`**. The local gcloud default is `omgrab-prod`, so always pass `--project=omgrab-dev` or set it.
- Data bucket: `gs://hendrik_mri_super_resolution/mri_denoiser/processed_data_mri/` (about 3.5 GiB). It has `full_res/train`, `low_res/{train,test}` and `samples/`, plus `low_res/train/episode_mapping.json` and `info.pt.bak_before_shape_cleanup`. It holds **processed data only**: no raw NIfTI and no code.
- Vertex artifacts bucket: `gs://omgrab-dev-diamond-vertex-artifacts/diamond/runs/` (about 55 runs). It contains:
  - `run-3d-*` with `checkpoints_3d/model_epoch_{10..50}.pt`. The paper used epoch 20; we have checkpoints up to 50.
  - `run-25d-10ep-*` with a copy of the processed data.
  - `run-swin2sr-2026030[7]-*` with `checkpoints_swin2sr/`. **An MRI-trained Swin2SR already exists**, but it was not reported in the paper and its training script is not in the repo.
  - Older `run-2026022x-*` runs and DIAMOND/game runs (`shorthands-*`) that have nothing to do with MRI.
- Training images are built by Cloud Build from `Dockerfile.vertex`, which does `COPY . /app`. So the images contain the exact local working tree that was used, including uncommitted files.
- Raw data: FOMO60K on HuggingFace (`FOMO-MRI/FOMO60K`). The paper used the `PT015_NKI` cohort, with 59 train subjects (100 sessions) and 5 test subjects (6 sessions, 993 sagittal slices).

## Repo state and known gaps

Recovered on 2026-10-23 from the code layer of the last Vertex training image, `us-central1-docker.pkg.dev/omgrab-dev/diamond-training/diamond-train:latest` (built 2026-03-07 08:17). `Dockerfile.vertex` does `COPY . /app`, so each image snapshots the working tree. All MRI-era images (2026-03-05 22:00 onward) were checked, and none has MRI code that the latest one lacks.
- `src/raw_data/scripts/main.py`: the MRI preprocessor (NIfTI to LR/HR pairs, 2D block-average downsampling). It had never been committed.
- `src/raw_data/scripts/view_mri.py`, `src/train_swin2sr.py`, `scripts/run_training_job_swin2sr.sh`.
- The `--mode swin2sr` support in `scripts/launch_vertex_training.py` and `scripts/submit_vertex_job.py`.
- `results/paper_v1/baseline_results.csv`: per-slice baseline metrics.

Unused DIAMOND code (Atari/CS:GO RL, world-model env, game UI, MKV pipeline) was removed on the same day. The paper's 2.5D checkpoint loads strictly and gives bit-identical samples before and after the removal.

Facts worth knowing:
- **Both models train on the same sagittal 2D-mode data** (`processed_data_mri`). `train_3d.py` stacks the slices back into volumes. The preprocessor's `--mode 3d` writes a `volume_x` HDF5 key that `dataset_3d.py` cannot read; it was never used.
- The 2.5D sampler config now lives under `sampler:` in `config/trainer_mri.yaml`. It used to be `world_model_env.diffusion_sampler_upsampling`. Defaults: 1-step stochastic Heun.
- `common.seed` is now honoured. Before, it was ignored and a random seed was always drawn.
- `state.pt` files from runs made before the cleanup cannot be resumed with `common.resume=True` (their trainer state has more fields). Warm-starting from their `agent_epoch_*.pt` through `initialization.path_to_ckpt` still works.
- The paper's 2.5D runs (`run-25d-10ep-lr1e-4`, `run-25d-10ep-lr5e5`) include `main.log` and W&B `.wandb` files with per-epoch logs. These are usable for the convergence answer (R1.2a). The 3D runs only have checkpoints (epochs 10 to 50).
- The 35.82 vs 35.67 dB mismatch (R1.4) has a known cause: `scripts/evaluate_25d.py` averages over all pooled slices, while Table 2 reports per-subject means.

Still open:
1. **No committed subject split.** The train/test subject IDs only exist implicitly, in the processed data and `low_res/train/episode_mapping.json`.
2. **No validation split and no validation logging.** `src/train_3d.py` logs only the training loss per epoch, so convergence cannot be shown.
3. **No EDSR fine-tuning script.** Only Swin2SR was trained on MRI.

## Reviewer comment triage

Legend: **MUST** = required experiment · **SHOULD** = high value, run if budget allows · **TEXT** = manuscript edit only · **REBUT** = explain in the response letter, no experiment.

| # | Comment (reviewers) | Decision | Action |
|---|---|---|---|
| 1 | Test set too small (5 subj); 59 train subjects not justified (R1.5, R2.1, R2.3, R3.1, R4, R5) | **MUST** | Larger FOMO60K train/val/test split at the subject level (see E1). Explain why the subset size was chosen. |
| 2 | Unequal sampling: 3D uses 20-step Euler, 2.5D uses 1-step Heun (R1.2, R2-m2) | **MUST** | Run both models with the same 20-step sampler (E2). The step sweep is part of E3. |
| 3 | No ablations: steps, patch size, attention, P_mean/P_std, context slices (R1.2c, R1.6, R5) | **MUST** | Small ablation (E3). |
| 4 | Unequal epochs, convergence not shown (R1.2a, R2-m1) | **MUST** | Validation loss and PSNR/SSIM curves vs. training steps for **every** trained model (E4). |
| 5 | Synthetic block-averaging degradation; no k-space, noise or Gibbs (R1.3, R2.4, R5) | **MUST** | k-space truncation degradation (E5). State the clinical scope in the abstract and conclusion. |
| 6 | No diffusion / MRI-specific SR comparator (R1.1, R3.6, R4) | **MUST** | Add at least one more diffusion model (E6) and discuss it at the algorithm level vs. Xia et al. and Wu et al. (R4). |
| 7 | Baselines use DIV2K weights, not MRI fine-tuning (R1.1, R2.6, R4, R5; raised by every reviewer) | **MUST** (added) | Retrain/fine-tune EDSR and Swin2SR on the same MRI training set and degradation (E7). Swin2SR runs already exist in the bucket. |
| 8 | No CIs, no significance tests, slices not independent (R1.5, R3.2, R5) | **MUST** (added, cheap) | Subject-level mean ± SD, bootstrap 95% CI over subjects, paired Wilcoxon at the subject level (E8). Uncertainty bands on per-slice plots. |
| 9 | 35.82 vs 35.67 dB inconsistency (R1.4) | **MUST** (added) | Define one aggregation method (subject-level mean) and recompute every table from a single eval script. |
| 10 | Single dataset; external validation on IXI/HCP (R2.2, R4, R5) | **SHOULD** | Zero-shot evaluation on IXI T1 (public, small) (E9). |
| 11 | Hallucination / anatomical fidelity / downstream task (R2.7, R3.3, R5) | **SHOULD** | SynthSeg (or FastSurfer) segmentation Dice, SR vs. GT; error maps; a failure-case figure (E10). |
| 12 | Compute cost, memory, inference time; hardware not consistent (R3.9, R5-minor) | **SHOULD** (cheap) | Table of params, peak GPU memory, time per volume, all on the same GPU. |
| 13 | Repeated training runs / seeds (R5) | **SHOULD** | 3 seeds for the final 3D and 2.5D models if budget allows. Otherwise acknowledge as a limitation. |
| 14 | Other MRI-specific non-diffusion SR (mDCSRN, SynthSR) (R1.1) | **SHOULD** | mDCSRN trained on our split if time allows. SynthSR solves a different task: discuss it, don't benchmark it. |
| 15 | Through-plane SR is more clinically relevant (R1-m7) | Stretch / TEXT | Could reuse the E5 pipeline along the slice axis. Otherwise state it as a limitation. |
| 16 | T1-only (R1-m8) | Stretch / TEXT | FOMO60K includes other contrasts. Zero-shot T2/FLAIR evaluation only if cheap. |
| 17 | Explain why 3D beats 2.5D (R3.5) | TEXT | Use ablation results (context slices, steps) to support the argument. |
| 18 | Limited novelty; DIAMOND origin (R3.4, R5, R1-m6, R1-m9) | TEXT | Frame the contribution as a controlled comparison. Document every change made relative to DIAMOND (channels, attention, sigma). Name the DIAMOND origin in the abstract/introduction. |
| 19 | Incomplete setup description (R3.8) and validation/model selection (R2-m1) | TEXT | Full degradation, training and config details in the paper and supplement. |
| 20 | Fig. 1 and Fig. 4 captions, ref [8] uncited, abstract phrasing (R1-m1, m2, m3, m10) | TEXT | Fix directly. |
| 21 | LPIPS validity on grayscale MRI; 2.5D SSIM below baselines (R1-m4, m5) | TEXT | Add a limitation note and an explicit discussion. |
| 22 | "Super-resolution" is a controlled synthetic setting (R2.5, R2-m3) | TEXT | Reword the introduction, discussion and conclusion. |
| 23 | Missing MRI-SR related work (R1 Q4, R2 Q4) | TEXT | Expand related work. |
| 24 | Radiologist reader study (R2.7, R5) | REBUT | Out of scope; offer the segmentation-based downstream metric (row 11) instead. |
| 25 | "Ecosystem-level consequences of warming" (R2.7) | REBUT | Clearly pasted in by mistake from another review. Point this out politely. |
| 26 | Pathological cases (R1 Q3) | TEXT | State as a limitation. |

## Experiments

General rules for every experiment:
- Fixed, committed subject-level split manifests under `splits/` (train / val / test, subject IDs).
- Seeds logged.
- Config and git SHA saved with each run.
- Every metric computed by **one** evaluation script.
- Results saved as JSON/CSV in the run directory and pulled into `results/`.

- **E1 — Larger dataset.**
  - Draw more subjects from FOMO60K, starting with NKI and possibly other T1 cohorts.
  - Split at the subject level. Target: test ≥ 30 subjects, plus a separate val split (about 10 subjects) for model selection.
  - Record the reason for the final size (compute or storage).
  - Generate train, val and test data with the recovered preprocessing script.
- **E2 — Matched sampler.** Use the same EDM sampler (Euler or Heun; pick one and justify) with 20 steps for both 3D and 2.5D. Report the old 1-step 2.5D result for reference.
- **E3 — Small hyperparameter ablation.** Use a reduced budget (subset of train data, fewer steps) and the val split.
  - Sampling steps {1, 5, 20} for both models.
  - 2.5D context slices: {1 previous, both neighbours, 2 on each side}.
  - 3D patch size: {32³, 64³}.
  - EDM sigma distribution (P_mean, P_std): {(-1.2, 1.2), (-0.6, 1.0), (0.0, 1.0)}.
  - Attention: deepest level only vs. the two deepest levels (only if budget allows).
  - One factor at a time around the default. Table plus one figure.
- **E4 — Convergence.**
  - Add periodic validation (loss, PSNR, SSIM on a fixed val subset) to the 2.5D trainer (`src/trainer.py`), the 3D trainer (`src/train_3d.py`) and every baseline trainer.
  - Log to W&B and CSV.
  - Train each model until val PSNR plateaus, with the same budget in GPU-hours or optimizer steps.
  - Plot curves for all models. Model selection uses the best val checkpoint.
- **E5 — k-space degradation.**
  - Implement `degrade_kspace(vol, factor, axes, noise_sigma)`: FFT, centre-crop or zero the outer k-space (truncation; produces Gibbs), add optional complex Gaussian noise, inverse FFT, take the magnitude.
  - (a) Evaluate the models trained with block averaging on a k-space-degraded test set (robustness).
  - (b) Retrain the main models with k-space degradation and evaluate them on both test sets.
  - Optional: through-plane variant.
- **E6 — Additional diffusion model.**
  - Candidates, picked by availability of reproducible public code:
    - SR3-style conditional DDPM (standard reference).
    - ResShift (efficient residual-shifting diffusion for SR).
    - Xia et al.'s partial diffusion model (explicitly named by R1, already cited).
  - Prefer an MRI-specific one (Xia et al.) if its code exists. Otherwise use SR3 or ResShift.
  - Train it on the same split and degradation, and include it in E4 and E8.
- **E7 — Fair baselines.** EDSR and Swin2SR trained/fine-tuned on the same data and degradation. Bicubic stays as a reference.
- **E8 — Statistics.**
  - Per-subject metrics, mean ± SD, bootstrap 95% CI over subjects.
  - Paired Wilcoxon signed-rank between methods (subject level, with Holm correction).
  - Per-slice plots with bands.
- **E9 — External validation.** IXI T1 with the same preprocessing and zero-shot inference.
- **E10 — Fidelity / downstream.** SynthSeg on GT vs. SR volumes, Dice per structure; failure cases; difference maps.

## Phases

0. **Recovery & reproducibility.**
   - Recover the MRI preprocessing script, the Swin2SR training code and the subject lists from: the Vertex training images in Artifact Registry (`omgrab-dev`), Vertex custom-job specs and Cloud Logging, the run directories, the bucket metadata (`episode_mapping.json`, `info.pt*`) and any local copies.
   - Commit what we recover.
   - Re-run the evaluation of the existing epoch-20 3D and 10-epoch 2.5D checkpoints, and check they reproduce the paper's numbers.
1. **Data.** E1 splits, plus the k-space test set (E5a) and IXI (E9).
2. **Tooling.** Validation logging (E4), a single eval harness with statistics (E8), compute profiling, and the `degrade_kspace` function.
3. **Main training.** 3D, 2.5D, EDSR, Swin2SR and the new diffusion model on the new split.
4. **Ablations.** E3, and E5b.
5. **Analysis.** E10, figures, tables.
6. **Writing.** Update `preprint_arxiv.tex`, prepare the highlighted-changes PDF, and write the response-to-reviewers document (one entry per comment, matching the triage table).

## Conventions

- Training on Vertex AI uses `scripts/launch_vertex_training.py`. Write new runs to `gs://omgrab-dev-diamond-vertex-artifacts/diamond/runs/<descriptive-name>/`.
- Never overwrite the original paper runs or the processed data in the bucket. Write new processed data under a new prefix, e.g. `processed_data_mri_v2/`.
- Do not commit data, checkpoints or `.nii` files (already covered by `.gitignore`).
- Keep the triage table above up to date as decisions change.
