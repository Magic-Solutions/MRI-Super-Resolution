# CLAUDE.md — MRI Super-Resolution: resubmission to IEEE Access

## Goal

This repo holds the code for the manuscript *"Comparative Analysis of 3D Convolutional and 2.5D Slice-Conditioned U-Net Architectures for MRI Super-Resolution via Elucidated Diffusion Models"* (IEEE Access, Manuscript ID Access-2026-10274). The paper was **not accepted** (decision 7 May 2026). The editor encouraged a resubmission.

**IEEE Access allows only ONE resubmission.** If the revised paper does not address *every* reviewer concern, it is rejected for good. So for each comment we must either (a) run the experiment, (b) change the text, or (c) give a reasoned rebuttal in the response letter.

The work has two parts:
1. **Triage** every reviewer comment: decide whether to address it with experiments, with text changes, or with a rebuttal. See the triage table below.
2. **Run** the selected experiments, then update the manuscript and write the response-to-reviewers document. That document needs, for each comment: the concern, our response, and the action taken.

Sources:
- Reviewer comments: `publication/reviewer-comments.html` (5 reviewers). R5 writes "Technical Appendix attached Access-2026-10274_Proof_hi_flaws.pdf", but no such file was attached: Eshaan checked the IEEE portal on 2026-10-25. The review text is all we have.
- Manuscript: `paper.tex` (`\documentclass{ieeeaccess}`), the file to resubmit. It lives in Overleaf (project 699cae01a03f708c8f9aaf8d); a LaTeX export is at `../context/manuscript/extracted/manuscript_ieee_access/paper.tex` on Eshaan's Mac. The repo has only the arXiv version, `publication/manuscript_ieee_access/preprint_arxiv.tex`.

## Infrastructure

- GCP project: **`omgrab-dev`**. The local gcloud default is `omgrab-prod` (on Eshaan's Mac it is `coastal-burner-331405`), so always pass `--project=omgrab-dev` or set it.
- Region: all 68 past Vertex custom jobs ran in `us-central1` on `g2-standard-8` with 1× L4. MRI jobs used the service accounts `diamond-vertex-trainer@omgrab-dev.iam.gserviceaccount.com` (2.5D/3D) and `omgrab-dev-jobs@omgrab-dev.iam.gserviceaccount.com` (Swin2SR), and read W&B from the secret `wandb-api-key`. Cloud Logging keeps only 30 days, so the March job logs are gone.
- Data bucket: `gs://hendrik_mri_super_resolution/mri_denoiser/processed_data_mri/` (about 3.5 GiB). It has `full_res/train`, `low_res/{train,test}` and `samples/`, plus `low_res/train/episode_mapping.json` and `info.pt.bak_before_shape_cleanup`. It holds **processed data only**: no raw NIfTI and no code.
- Vertex artifacts bucket: `gs://omgrab-dev-diamond-vertex-artifacts/diamond/runs/` (55 folders on 2026-10-24: 11 from the MRI project, 44 unrelated). It contains:
  - `run-3d-*` with `checkpoints_3d/model_epoch_{10..50}.pt`. The paper reports the 3D model after 20 epochs. The published `model_3d_epoch_20.pt` is `run-3d-20260306-082303/checkpoints_3d/model_epoch_20.pt` (see "Published weights" below).
  - `run-25d-10ep-*` with a copy of the processed data.
  - `run-swin2sr-2026030[7]-*` with `checkpoints_swin2sr/`. **An MRI-trained Swin2SR already exists**, but it was not reported in the paper and its training script is not in the repo.
  - Older `run-2026022x-*` runs and DIAMOND/game runs (`shorthands-*`) that have nothing to do with MRI.
- Training images are built by Cloud Build from `Dockerfile.vertex`, which does `COPY . /app`. So the images contain the exact local working tree that was used, including uncommitted files.
- Raw data: FOMO60K on HuggingFace. `FOMO-MRI/FOMO60K` now redirects to **`FOMO-MRI/FOMO50K`**, which is gated: accept the terms on the website, then run `huggingface-cli login`. `PT015_NKI` has 1,316 subjects and 2,281 sessions; 1,315 subjects (2,271 sessions) have `sub_<id>/ses_<n>/t1.nii.gz`. The NKI `LICENSE.txt` points to the Rockland Sample data usage agreement. The paper used the `PT015_NKI` cohort, with 59 train subjects (100 sessions) and 5 test subjects (6 sessions, 993 sagittal slices).
- **New resubmission dataset (E1, built 2026-10-24):** `gs://hendrik_mri_super_resolution/mri_denoiser/processed_data_mri_resub/`, 1,160 objects, 8.35 GB. Details are in the E1 section below.

## Repo state and known gaps

Recovered on 2026-10-23 from the code layer of the last Vertex training image, `us-central1-docker.pkg.dev/omgrab-dev/diamond-training/diamond-train:latest` (built 2026-03-07 08:17). `Dockerfile.vertex` does `COPY . /app`, so each image snapshots the working tree. All MRI-era images (2026-03-05 22:00 onward) were checked, and none has MRI code that the latest one lacks.
- `src/raw_data/scripts/main.py`: the MRI preprocessor (NIfTI to LR/HR pairs, 2D block-average downsampling). It had never been committed.
- `src/raw_data/scripts/view_mri.py`, `src/train_swin2sr.py`, `scripts/run_training_job_swin2sr.sh`.
- The `--mode swin2sr` support in `scripts/launch_vertex_training.py` and `scripts/submit_vertex_job.py`.
- `results/paper_v1/baseline_results.csv`: per-slice baseline metrics.

Unused DIAMOND code (Atari/CS:GO RL, world-model env, game UI, MKV pipeline) was removed on the same day. The paper's 2.5D checkpoint loads strictly and gives bit-identical samples before and after the removal.

Facts worth knowing:
- **Both models train on the same sagittal 2D-mode data** (`processed_data_mri`). `train_3d.py` stacks the slices back into volumes. The preprocessor's `--mode 3d` writes a `volume_x` HDF5 key that `dataset_3d.py` cannot read; it was never used.
- The 2.5D sampler config now lives under `sampler:` in `config/trainer_mri.yaml`. It used to be `world_model_env.diffusion_sampler_upsampling`. Defaults: 1 step, `order: 2` (Heun), σ_max 5, σ_min 1, `s_churn` 10. **This is not a Heun step.** With 1 step, `build_sigmas` gives σ = [5, 0], and `diffusion_sampler.py` skips the Heun correction when the next σ is 0. So the paper's "1-step Heun" (paper.tex lines 239, 283, 511, 515) is one Euler step with churn. `scripts/evaluate_25d.py` overrides the step count with `--num-steps`, which defaults to 20 (also at 371c36a), so which step count produced the paper's 2.5D numbers is **NOT VERIFIED**.
- **3D patch size: 64³ (VERIFIED 2026-10-25).** The image all three March 3D jobs ran (`diamond-train@sha256:b6f4be50…`, `:latest` from 2026-03-06 03:23 UTC until 2026-03-07) has `--patch-size` default 64 in `train_3d.py`, and Hendrik confirmed he passed no patch size. The rest of this bullet is the original reasoning: In `train_3d.py`, `--patch-size` is the LR patch (HR is `p × 2p × 2p`) and defaults to 64. `scripts/run_training_job_3d.sh` passes no `--patch-size`, and the March 3D job specs pass only `--epochs`. So the March 3D model most likely trained on 64³ LR patches (64×128×128 HR). `scripts/run_inference_3d.py` defaults to 32, and the paper (line 223) says 32³ LR. No surviving log confirms the training value.
- **Published weights: which March checkpoints the paper used (VERIFIED 2026-10-24).** The paper (line 49) links https://huggingface.co/Chichonnade/Comparative-Analysis-3D-2.5D-MRI-Super-Resolution-EDM (public; created 2026-03-06 20:52 UTC, last modified 2026-03-07 00:15 UTC). Each file's MD5 equals the GCS MD5 of exactly one of the 1,102 March `.pt` objects, and its sha256 matches the Hub's LFS id:

  | Published file | March checkpoint | Model card says |
  |---|---|---|
  | `agent_epoch_00010.pt` (2.5D) | `run-20260306-035738` epoch 10 (batch 4 × 8, lr 1e-4) | 10 epochs, 33.68 dB / 0.967, "single-step Heun" |
  | `model_3d_epoch_20.pt` | `run-3d-20260306-082303` epoch 20 (a 50-epoch run) | not described |
  | `model_final.pt` | `run-3d-20260306-052515` final (the 10-epoch run) | "3D EDM Upsampler", 10 epochs, 37.77 dB / 0.996 |

  - The paper's 2.5D model is therefore the 10-epoch run; the `run-25d-10ep-lr1e-4/-lr5e5` continuations were not published.
  - The paper reports the 3D model after 20 epochs (37.75 dB), which fits `model_3d_epoch_20.pt`; the card instead describes the 10-epoch `model_final.pt` at 37.77 dB. Which file produced the paper's Table 1 is not settled by the hashes alone.
  - For the same 2.5D checkpoint the card gives 33.68 dB and the paper 35.82 dB (Table 1). Possible causes, NOT VERIFIED: step count (the card's 1 step vs `evaluate_25d.py`'s default 20), test set, or aggregation.
  - The card says both models trained on 59 subjects; the 3D code reads all 258 sessions (156 subjects), NOT VERIFIED by a log.
- **How the first submission's numbers were produced (VERIFIED 2026-10-24).** The paper's test set was rebuilt from FOMO with `src/raw_data/scripts/main.py`: sub_2642 ses_1, sub_2952 ses_1, sub_6794 ses_1 and ses_2, sub_8390 ses_1, sub_9320 ses_1; 171+173+152+158+165+174 = 993 slices, since `process_2d` drops all-zero HR slices. Its per-slice interpolation PSNRs match `results/paper_v1/baseline_results.csv` to 5e-6 dB. It is in `gs://hendrik_mri_super_resolution/mri_denoiser/processed_data_mri_paper_test/`.
  - **The "Bicubic" baseline is bilinear.** The evaluation ran on MPS, where the scripts silently swap bicubic for bilinear. Bilinear gives the paper's 33.89 dB (Table 1) and its per-volume Table 2 values exactly. True bicubic: 35.65 dB, SSIM 0.973 (pooled, clamped).
  - **The 2.5D numbers used mismatched conditioning.** Training ran on CUDA with bicubic-upsampled conditioning (`Denoiser.forward`); the paper's evaluation ran on MPS with bilinear. Hendrik confirmed both on 2026-10-24: "I ran the eval on MPS locally on the test set (not on cuda)" and "the full training for both 2.5D and 3D was trained on cuda (not local mps)". Published checkpoint, legacy 1-step sampler, 993 slices: MPS/bilinear 35.82 dB / 0.9706, which reproduces Table 1 and Table 2 to within 0.01 dB per volume. CUDA/bicubic: 38.38 dB / 0.9837.
  - **Table 2 omits sub_9320.** Table 1 is the mean over all 993 slices; Table 2's averages are means of 5 per-volume means without sub_9320. That fully explains 35.82 vs 35.67 (R1.4).
  - **The paper's 3D model is `model_3d_epoch_20.pt`.** Re-evaluated on CUDA (20 steps, patch 32, 2-D SSIM), its per-volume PSNRs 37.44, 38.16, 37.34, 37.74, 38.11 match Table 2 (37.44, 38.15, 37.33, 37.74, 38.10). `model_final.pt` (the 10-epoch run) gives different values; its 6-volume mean, 37.77, is the model card's 3D figure.
  - **Table 1 averaged the two models differently.** Its 3D 37.75 is the mean of Table 2's 5 volume means; the 2.5D 35.82 is pooled over all 6 volumes.
  - **Matched comparison on the paper's test set** (CUDA, pooled over 993 slices, 2-D SSIM; run `mri-resub-eval-paper-20261024-2140`):

    | Method | PSNR | SSIM | LPIPS |
    |---|---|---|---|
    | Bilinear (the paper's "Bicubic") | 33.89 | 0.957 | 0.091 |
    | Bicubic (true) | 35.59 | 0.973 | 0.070 |
    | EDSR, DIV2K weights | 35.57 | 0.977 | 0.024 |
    | Swin2SR, DIV2K weights | 35.50 | 0.978 | 0.024 |
    | 2.5D (published), legacy sampler, 1 step | 38.38 | 0.984 | 0.022 |
    | 2.5D (published), legacy sampler, 20 steps | 33.32 | 0.914 | 0.134 |
    | 3D `model_3d_epoch_20.pt`, 20 steps, patch 32 | 37.84 | 0.985 | 0.020 |

    EDSR and Swin2SR reproduce the paper exactly; they take the LR image directly, so the device did not matter for them.
  - **3D inference patch size matters.** `evaluate_sr.py`, run `mri-resub-eval-paper-3dpatch-20261025-0015`: the 3D model evaluated with the 64³ patches it trained on (verified, see "3D patch size") gains 0.5 dB over the paper's 32³. With every model evaluated as trained (pooled over 993 slices; foreground PSNR counts only non-zero HR pixels):

    | Method | PSNR | PSNR foreground | SSIM | LPIPS |
    |---|---|---|---|---|
    | 2.5D, shared sampler, 20 steps | 38.69 | 32.75 | 0.9846 | 0.0183 |
    | 2.5D, legacy, 1 step | 38.38 | 32.70 | 0.9838 | 0.0219 |
    | 3D, patch 64 | 38.34 | 33.17 | 0.9851 | 0.0174 |
    | 3D, patch 32 (as in the paper) | 37.83 | 33.30 | 0.9848 | 0.0204 |
    | Bicubic | 35.65 | 29.90 | 0.9732 | 0.0704 |

    Paired per volume, 2.5D (shared, 20 steps) vs 3D (patch 64): whole-slice PSNR favours 2.5D on 5 of 6 volumes (+0.29 dB mean, Wilcoxon p = 0.063); foreground PSNR (−0.42 dB), SSIM (−0.0005) and LPIPS (+0.0009) favour 3D on 6 of 6 (p = 0.031 each, the smallest p a 6-volume test allows; 0.125 after Holm over the 4 metrics). So 3D is consistently, if slightly, better on the anatomy, and 2.5D's whole-slice lead comes from the background. The paper's "3D best on all metrics by 1.93 dB" does not hold, but neither does "2.5D beats 3D". The two models were also trained on different data (100 vs 258 sessions). Answer from the new runs, and report foreground PSNR alongside whole-slice PSNR.
  - **Sampler:** the legacy stochastic sampler (churn, σ_min 1) degrades with more steps. The shared deterministic sampler does not: published 2.5D on sub_2642, 1 / 5 / 20 steps = 38.01 / 38.30 / 38.31 dB. So E2's shared 20-step sampler is fair to the 2.5D model.
  - The model card's 2.5D figure (33.68 dB / 0.967) matches neither setting; it is closest to the legacy 20-step result (33.32 / 0.914). Its origin is NOT VERIFIED and does not matter for the resubmission.
- **Epochs are not comparable between the models.** A 2.5D "epoch" is a fixed `steps_per_epoch` = 400 optimizer steps (3,200 batches of 4 segments, 2 target slices each), not a pass over the data. A 3D or baseline epoch is a pass over the training volumes. Report optimizer steps or GPU-hours in the resubmission.
- `common.seed` is now honoured. Before, it was ignored and a random seed was always drawn.
- `state.pt` files from runs made before the cleanup cannot be resumed with `common.resume=True` (their trainer state has more fields). Warm-starting from their `agent_epoch_*.pt` through `initialization.path_to_ckpt` still works.
- The 2.5D runs `run-20260306-035738`, `run-25d-10ep-lr1e-4` and `run-25d-10ep-lr5e5` have per-epoch logs in their W&B files (see "The 2.5D run folders" below; `main.log` is empty). These are usable for the convergence answer (R1.2a). The 3D runs only have checkpoints (epochs 10 to 50).
- The 35.82 vs 35.67 dB mismatch (R1.4) has a known cause: `scripts/evaluate_25d.py` averages over all pooled slices, while Table 2 reports per-subject means.
- **The March models did not see the same training volumes.** `processed_data_mri/low_res/train` holds 258 episodes (156 subjects), but its `info.pt` lists only the first 100 (59 subjects, the paper's number). `info.pt.bak_before_shape_cleanup` lists all 258, with lengths {166, 167, 169, 176, 254}. Who reads what:
  - **2.5D (`src/trainer.py`):** reads `info.pt`. Its logs say "train_dataset: 100 episodes, 17600 steps" and "test_dataset: 0 episodes".
  - **Swin2SR (`src/train_swin2sr.py`):** iterates `lr_dataset.num_episodes`, i.e. `info.pt`, so 100 episodes.
  - **3D (`src/data/dataset_3d.py::_discover_volume_pairs`):** reads `episode_mapping.json`, so all 258 episodes (156 subjects), including the non-176-slice volumes. This is from the code; the March 3D logs have expired, so there is no log confirming it.
  - The paper's 3D vs 2.5D comparison is therefore also confounded by training-set size. The new dataset fixes this: its `info.pt` and `episode_mapping.json` list the same volumes.
- **No validation was ever run.** `low_res/test/info.pt` is empty, and the 2.5D runs show `Evaluating upsampler: 0it` every epoch. Model selection had no validation data (R2-m1).
- **The paper's test subjects** (sub_2642, sub_2952, sub_6794 ses1+ses2, sub_8390) are not stored anywhere in GCS. They can be regenerated from FOMO.
- **Preprocessing is verified bit-exact.** Raw FOMO `t1.nii.gz` is int16, RAS, (176, 256, 256), 1.0×0.977×0.977 mm, sagittal along axis 0, with no resample/crop/pad. The rule:
  - `p1, p99 = np.percentile(vol[vol > 0], [1, 99])`
  - clip, then `(x − p1)/(p99 − p1)·255`, then `astype(uint8)` (truncates)
  - HR frame i = `np.rot90(x[i])`
  - LR = floor of the 2×2 block mean
  This reproduces the stored HR and LR exactly (100%, max diff 0) for sub_8112 ses_1 and sub_10915 ses_1, and matches `src/raw_data/scripts/main.py`. `main.py` computes in float32; the new scripts use float64, and both gave identical output on these volumes.
- **The 2.5D run folders:** `main.log` is 0 bytes. The per-epoch stdout is in `wandb/run-*/files/output.log`; the scalars are in `wandb-summary.json` and the `.wandb` files.
- **Reference errors in the manuscript** (checked against publisher/arXiv pages):
  - [11]: wrong authors. The actual authors are Jannat et al., Front. Hum. Neurosci. 2025.
  - [12]: wrong authors and venue details. The actual entry is Zhao et al., "MRI Super-Resolution With Partial Diffusion Models", IEEE TMI 44(3), 2025, doi 10.1109/TMI.2024.3483109 (prostate/knee, not brain). The text, and E6 below, call it "Xia et al.".
  - [13]: wrong authors. The actual authors are Safari et al., Res-SRDiff, arXiv 2503.01576. The text calls it "Wu et al.".
  - [19]: FOMO60K is now `FOMO-MRI/FOMO50K`, and the dataset card asks users to cite Cerri et al., arXiv:2506.14432.
  - [23]: "M. Burber" should be M. Burchi.

Still open:
1. ~~No committed subject split.~~ The new dataset ships a `split.json` (see E1). It is not committed under `splits/` yet.
2. **Validation logging: done for every model, not committed** (on `resub/eval-fixes-main`).
   - **3D:** patch 0001 gives `src/train_3d.py` a per-epoch validation loss, periodic validation PSNR/SSIM and `model_best_val.pt` (item 4).
   - **2.5D** (`src/trainer.py`, 2026-10-24):
     - The validation loss (`test_upsampler`) now runs under a fixed seed, so it is comparable across epochs.
     - `trainer_mri.yaml` now validates every epoch (`evaluation.every: 1`, was 20) over the whole validation split (`max_test_batches: null`, was 50).
     - New `validate_upsampler_sr` samples 16 evenly spaced slices per validation volume with the shared 20-step EDM sampler (as the 3D validation does), seeded, and logs `upsampler/val/psnr` and `ssim`.
     - The best epoch by validation PSNR is saved as `checkpoints/agent_best_val.pt`. Every validation is appended to `checkpoints/val_metrics.csv`, which `run_training_job.sh` uploads with the checkpoints.
     - Smoke-tested on mini with a small U-Net: 2 epochs, and a rerun gives an identical `val_metrics.csv`. The full 51M model trains at about 3 min per step on the Mac's MPS backend, so it was only started there, not finished.
   - **Baselines:** see item 3.
3. **EDSR and Swin2SR now train from one script, not committed.** `src/train_swin2sr.py --arch {swin2sr,edsr}`. EDSR-base (16 residual blocks, 64 features, 1.37M parameters) is built from scratch with `super_image`'s `EdsrConfig`, so both baselines see the same data, L1 loss and validation. On Vertex, EDSR uses the existing mode: `launch_vertex_training.py --mode swin2sr --train-swin2sr-args "--arch edsr"`. Other changes to the script:
   - It stores slices as uint8: 7.2 GB for the 88,000 training slices instead of 28.8 GB as float32. The old version would probably not have fit in a g2-standard-8's 32 GB. Items are the same 0–255 values as before (LR differs by ≤1.5e-8 float rounding).
   - Validation now runs every epoch (was 5) on 16 evenly spaced slices of every validation volume. Before, it used the first 200 slices, about one subject. LPIPS is now opt-in (`--eval-lpips`).
   - It saves `<arch>_best_val.pt` and `val_metrics.csv`, and sets a seed (`--seed`, default 0).
   - Both were smoke-tested on mini (64 training slices, 2 epochs): EDSR on MPS, Swin2SR on CPU (it barely moves on MPS). Both write `<arch>_best_val.pt` and `val_metrics.csv`.
4. **Patches 0001 and 0002 are rebased and smoke-tested, but not committed** (2026-10-24). They sit on the local branch `resub/eval-fixes-main` (from main @ 745b582) as staged, uncommitted changes in 10 files. The `info.pt` fix in `src/data/dataset.py` (item 5) is a further unstaged change. The only conflict was the `DiffusionSampler` import in `scripts/evaluate_25d.py`, resolved to main's `models.diffusion`. The applied `make_split.py`, `preprocess_fomo.py`, `edm_sampler.py` and `patchwise_3d.py` are byte-identical to the standalone copies. Smoke tests on `~/data/mri_resub/mini/` (Mac, MPS, torch 2.4.1 and numpy 1.26 as pinned):
   - `train_3d.py` with validation: 2 epochs, small U-Net, 4 train and 2 val volumes. It logs train and val loss, val PSNR/SSIM, and saves `model_best_val.pt`.
   - `run_inference_3d.py` on the 2 test volumes: every SSIM call now gets a 2-D slice (704 of 704).
   - The patched `Denoiser3d.sample` is bit-identical to main's at 1, 5 and 20 steps.
   - `evaluate_25d.py --sampler legacy` and `--sampler shared` both run end to end on a randomly initialised 2.5D agent.
   - `make_split.py` on mini: the 2.5D loader (`info.pt`) and the 3D loader (`episode_mapping.json`) see the same volumes in every split.
   - Not tested: `preprocess_fomo.py` (it needs a FOMO download; it was tested on synthetic NIfTIs before), the 2.5D trainer, Vertex.
   - Local quirks, not code bugs: on MPS the 3D model needs `PYTORCH_ENABLE_MPS_FALLBACK=1` (`upsample_trilinear3d`). LPIPS was stubbed out, because it downloads AlexNet weights on first use.
5. **Fixed, not committed: the new dataset's `info.pt` files didn't load under numpy 1.26** (2026-10-24, on `resub/eval-fixes-main`, unstaged). `Dataset.load_from_default_path` in `src/data/dataset.py` now unpickles `info.pt` with a pickle module whose Unpickler maps `numpy._core.*` to `numpy.core.*` when numpy 1.x is installed; nothing else changes. Checks:
   - `evaluate_25d.py` run directly loads the mini test set under numpy 1.26, then runs end to end with `--sampler legacy` and `--sampler shared` (LPIPS stubbed only).
   - The loader reads numpy-2-written and numpy-1.26-written `info.pt` under numpy 1.26 and 2.2 (all four pass).
   - Not tested on Vertex.

   The background: the files were pickled under numpy 2.x and reference `numpy._core.multiarray`. `requirements.txt` pins `numpy==1.26.0`, and the Vertex image installs it (`Dockerfile.vertex`). With the pinned versions, `torch.load` fails with `ModuleNotFoundError: No module named 'numpy._core'` (reproduced locally; not yet run on Vertex).
   - Affected: the three `info.pt` files in `processed_data_mri_resub/`, so the 2.5D trainer, `train_swin2sr.py`, `evaluate_25d.py`, `evaluate_baselines.py` and `run_inference_25d.py`. Every one of them loads through `Dataset.load_from_default_path` in `src/data/dataset.py`.
   - Not affected: the episode `.pt` files (no numpy objects), the HDF5 files, and the 3D trainer (it reads `episode_mapping.json`).
   - An `info.pt` written under numpy 1.26 loads under both numpy 1.26 and 2.2 (tested).
   - Eshaan chose option (a), the remap on load, over (b) rewriting the files under a new GCS prefix and (c) moving to numpy 2.x. A global `sys.modules` alias was rejected because it breaks the torch and scipy imports.
6. **Evaluation: use `scripts/evaluate_sr.py` and `scripts/compare_sr.py` for every reported number** (E8). They do all interpolation on CPU, skip all-zero HR slices, clamp predictions, and add foreground PSNR, per-volume statistics, bootstrap CIs and Holm-corrected Wilcoxon tests. The older scripts stay for reproducing the first submission: on MPS, `evaluate_25d.py` and `evaluate_baselines.py` silently use bilinear for "bicubic", which is how the paper's numbers came about (see "How the first submission's numbers were produced"). Also, the `vol*_axial_mid.png` figures from `run_inference_3d.py` show a sagittal slice, because axis 0 is sagittal (a naming bug only). `evaluate_25d.py` used to set no seed, so repeated runs differed by up to 0.045 dB per slice. It now takes `--seed` (default 0), and two runs give identical CSVs. It also takes `--override` for Hydra overrides of the model config, which is needed to evaluate a both-neighbours model.
7. **Both-neighbours conditioning for 2.5D, not committed** (R1.2c, E3).
   - Set `agent.upsampler.inner_model.num_next_slices_conditioning=1`. Together with `num_steps_conditioning: 1`, the model is then conditioned on the previous and the next LR slice. The default 0 keeps the old model: same parameter shapes (a pre-change checkpoint loads strictly) and the same conditioning.
   - The frame order is `[previous…, next…, target LR]` everywhere. `src/models/diffusion/slice_conditioning.py` builds it for sampling: trainer validation, W&B samples and `evaluate_25d.py`.
   - Tested: with 0 next slices the helper equals the old inline code for every slice; the frames the training loss sees equal the frames the samplers build; 2-epoch smoke training works; `evaluate_25d.py` runs on it.
   - `run_inference_25d.py` and `generate_paper_figures.py` still build previous-only context, and now refuse a both-neighbours model with an assert.
   - With `num_next = 1`, the last slice of a volume is never a training target. BatchSampler pads only before the start. That is 1 slice in 176.
   - Consequence, found 2026-10-27 in the final evaluation: sampling filled the missing next neighbour of the last slice with a zero frame (mid-grey in [-1, 1]), which the model never saw in training. It put a uniform grey offset on that slice: 27.9 dB against 50.1 dB for the one-sided model, on all 70 test volumes. That cost about 0.13 dB of per-volume test PSNR and explains the 1.1 dB gap in its in-training validation (16 slices per volume, last one included). Fixed in 92cafdb: next neighbours past the end repeat the last slice; previous neighbours keep zero padding, as in training. Only the both-neighbours model is affected. Its checkpoint (epoch 27) was still selected by the buggy validation; all 30 epoch checkpoints are kept if re-selection is ever needed.
8. **Vertex GPU quota not checked.** `gcloud alpha services quota list` needs the gcloud `alpha` component, which is not installed. The Cloud Quotas API would need an access token in a request. Check it in the console (IAM & Admin → Quotas, service "Vertex AI API", region us-central1, "custom model training … L4"), or decide how to query it.

## Reviewer comment triage

Legend: **MUST** = required experiment · **SHOULD** = high value, run if budget allows · **TEXT** = manuscript edit only · **REBUT** = explain in the response letter, no experiment.

| # | Comment (reviewers) | Decision | Action |
|---|---|---|---|
| 1 | Test set too small (5 subj); 59 train subjects not justified (R1.5, R2.1, R2.3, R3.1, R4, R5) | **MUST** | Larger FOMO60K train/val/test split at the subject level (see E1). Explain why the subset size was chosen. |
| 2 | Unequal sampling: 3D uses 20-step Euler; 2.5D is described as "1-step Heun" (R1.2, R2-m2) | **MUST** | Run both models with the same 20-step sampler (E2). The step sweep is part of E3. The 2.5D "1-step Heun" is in fact one Euler step with churn (σ 5 → 0; see "Facts worth knowing"). Correct paper.tex lines 239, 283, 511 and 515, and say so in the R1.2 and R2-m2 answers. |
| 3 | No ablations: steps, patch size, attention, P_mean/P_std, context slices (R1.2c, R1.6, R5) | **MUST** | Small ablation (E3). Also add the sampling sigma schedule (sigma_min, sigma_max, rho) and the 3D patch overlap at inference (R5: "sigma schedule sensitivity", "patch overlap effects"). |
| 4 | Unequal epochs, convergence not shown (R1.2a, R2-m1) | **MUST** | Validation loss and PSNR/SSIM curves vs. training steps for **every** trained model (E4). Epochs are not comparable: a 2.5D epoch is a fixed 400 optimizer steps, while a 3D or baseline epoch is a pass over the data (and in March the 3D model saw 2.6× more volumes). So the resubmission reports optimizer steps or GPU-hours, not epochs. |
| 5 | Synthetic block-averaging degradation; no k-space, noise or Gibbs (R1.3, R2.4, R5) | **MUST** | k-space truncation degradation (E5). State the clinical scope in the abstract and conclusion. R5 also lists Gibbs ringing, motion and scanner noise: k-space truncation already produces Gibbs; add Rician noise as a cheap extra variant; state motion as a limitation. |
| 6 | No diffusion / MRI-specific SR comparator (R1.1, R3.6, R4) | **MUST** | Add at least one more diffusion model (E6) and discuss it at the algorithm level vs. Xia et al. and Wu et al. (R4). Note: the paper's "Xia et al. [12]" is actually Zhao et al., "MRI Super-Resolution With Partial Diffusion Models", IEEE TMI 2025, and its "Wu et al. [13]" is actually Safari et al., Res-SRDiff (arXiv 2503.01576). Use the correct names in the paper and in the response to R1 and R4. |
| 7 | Baselines use DIV2K weights, not MRI fine-tuning (R1.1, R2.6, R3.7, R4, R5; raised by every reviewer) | **MUST** (added) | Retrain/fine-tune EDSR and Swin2SR on the same MRI training set and degradation (E7). Swin2SR runs already exist in the bucket. R4: domain-trained models beating off-the-shelf ones is already well established, so once the baselines are retrained the claim must rest on the architecture comparison. |
| 8 | No CIs, no significance tests, slices not independent (R1.5, R3.2, R5) | **MUST** (added, cheap) | Subject-level mean ± SD, bootstrap 95% CI over subjects, paired Wilcoxon at the subject level (E8). Uncertainty bands on per-slice plots. |
| 9 | 35.82 vs 35.67 dB inconsistency (R1.4) | **MUST** (added) | Define one aggregation method (subject-level mean) and recompute every table from a single eval script (`evaluate_sr.py` + `compare_sr.py`). Cause, verified: Table 2 omits sub_9320 and averages volume means; Table 1 pools all 993 slices. |
| 10 | Single dataset; external validation on IXI/HCP (R2.2, R4, R5) | **SHOULD** | Zero-shot evaluation on IXI T1 (public, small) (E9). |
| 11 | Hallucination / anatomical fidelity / downstream task (R2.7, R3.3, R5) | **SHOULD** | SynthSeg (or FastSurfer) segmentation Dice, SR vs. GT; error maps; a failure-case figure (E10). |
| 12 | Compute cost, memory, inference time; hardware not consistent (R3.9, R5-minor) | **SHOULD** (cheap) | Table of params, peak GPU memory, time per volume, all on the same GPU. |
| 13 | Repeated training runs / seeds (R5) | **SHOULD** | 3 seeds for the final 3D and 2.5D models if budget allows. Otherwise acknowledge as a limitation. |
| 14 | Other MRI-specific non-diffusion SR (mDCSRN, SynthSR) (R1.1) | **SHOULD** | mDCSRN trained on our split if time allows. SynthSR solves a different task: discuss it, don't benchmark it. |
| 15 | Through-plane SR is more clinically relevant (R1-m7) | Stretch / TEXT | Could reuse the E5 pipeline along the slice axis. Otherwise state it as a limitation. |
| 16 | T1-only (R1-m8) | Stretch / TEXT | FOMO60K includes other contrasts. Zero-shot T2/FLAIR evaluation only if cheap. |
| 17 | Explain why 3D beats 2.5D (R3.5) | TEXT | **The premise does not hold as published:** evaluated as trained, the published 2.5D and 3D models are within about 0.5 dB, with whole-slice PSNR favouring 2.5D and foreground PSNR, SSIM and LPIPS slightly favouring 3D (see "How the first submission's numbers were produced"). Answer from the new runs, which give both models the same data. Use ablation results (context slices, steps) to support the argument. Note: in March the 3D model trained on 258 sessions (`episode_mapping.json`) while 2.5D trained on 100 (`info.pt`), from reading the code. The old gap is therefore confounded by data size; the new split gives both models the same data. |
| 18 | Limited novelty; DIAMOND origin (R3.4, R5, R1-m6, R1-m9) | TEXT | Frame the contribution as a controlled comparison. Document every change made relative to DIAMOND (channels, attention, sigma). Name the DIAMOND origin in the abstract/introduction. |
| 19 | Incomplete setup description (R3.8) and validation/model selection (R2-m1) | TEXT | Full degradation, training and config details in the paper and supplement. State the 3D patch size actually used: the paper says 32³ LR, but the March training most likely used 64³ (NOT VERIFIED). The new runs set `--patch-size` explicitly. |
| 20 | Fig. 1 and Fig. 4 captions, references, abstract phrasing (R1-m1, m2, m3, m10; R5-minor: captions should give a more anatomically precise interpretation) | TEXT | Fix directly. References: wrong author lists in [11], [12] and [13], plus issues in [19] and [23] (see `../context/reference_errors.md`). Six bibitems are never cited: b4, b5, b8, b14, b15, b16. |
| 21 | LPIPS validity on grayscale MRI; 2.5D SSIM below baselines (R1-m4, m5) | TEXT | Add a limitation note and an explicit discussion. The low 2.5D SSIM (0.971) came from evaluating with bilinear instead of the trained bicubic conditioning; with bicubic it is 0.984. Finding: `scripts/run_inference_3d.py` computed 3D SSIM on 5-D slices, so `metrics.ssim` used an 11×11×11 window over zero-padded depth, which inflates SSIM (0.817 became 0.976 on the same image pair). The paper's 3D SSIM of 0.997 is not comparable with the 2-D SSIM of 2.5D and the baselines. This is fixed in patch 0001; recompute all SSIM as 2-D per slice. |
| 22 | "Super-resolution" is a controlled synthetic setting (R2.5, R2-m3; R5-minor: "occasional overstatement of conclusions relative to dataset scale") | TEXT | Reword the introduction, discussion and conclusion. |
| 23 | Missing MRI-SR related work (R1 Q4, R2 Q4) | TEXT | Expand related work. |
| 24 | Radiologist reader study (R2.7, R5) | REBUT | Out of scope; offer the segmentation-based downstream metric (row 11) instead. |
| 25 | "Ecosystem-level consequences of warming" (R2.7) | REBUT | Clearly pasted in by mistake from another review. Point this out politely. |
| 26 | Pathological cases (R1 Q3) | TEXT | State as a limitation. |
| 27 | R5's technical appendix `Access-2026-10274_Proof_hi_flaws.pdf` (R5) | **REBUT** | Never attached (portal checked 2026-10-25), so the triage covers R5's review text only. In the response to R5, say politely that the appendix was not included with the decision and that every point in the review text is answered. |
| 28 | Submission package (decision letter) | **MUST** | Get the response template and the Resubmission Checklist (both attached to the letter, not in the repo). **The "Request for Byline Change" form is required**: Eshaan is joining the author list (decided 2026-10-24). It needs a detailed justification of each author's contribution, is uploaded as "Request for byline change form", and the justification is entered again during submission. The letter calls author changes "rare and exceptional" and leaves the decision to the Editor. Agree the author order and get every author's OK, including Ludovic Corcos and Logan Rouge. Owner: Hendrik (IEEE Author Portal); open questions are in `../ASK_HENDRIK.md`. |

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
  - **Status: DONE (2026-10-24).** Eshaan chose **500 train / 6 val / 70 test subjects**, matching DisC-Diff's IXI split.
    - Data: NKI only, one session per subject (the lowest), 176-slice volumes only.
    - Location: `gs://hendrik_mri_super_resolution/mri_denoiser/processed_data_mri_resub/`, laid out as follows.
      - `training/low_res/train`, `training/full_res/train`: 500 subjects, 88,000 slices.
      - `training/low_res/test`, `training/full_res/test`: the 6 **validation** subjects (211, 7657, 8027, 8917, 9095, 9530), 1,056 slices. They sit under `test` because the 2.5D trainer evaluates on `<data>/test` and `train_3d.py` takes `--lr-test/--hr-test`.
      - `evaluation/low_res/test`, `evaluation/full_res/test`: the 70 held-out **test** subjects, 12,320 slices. Never use them for training or model selection.
      - `split.json` (all subject and session IDs per split) and `sessions.txt` (the 650 sessions selected with seed 0).
    - 2.5D: `PREPROCESSED_DATA_URI=…/processed_data_mri_resub/training`. 3D: train/val from `training/`, final test from `evaluation/`.
    - **Its `info.pt` files were pickled under numpy 2.x.** Under the pinned numpy 1.26 they load only through the fixed `Dataset.load_from_default_path` (see "Still open", item 5; fixed but not committed).
    - Built by `scripts/preprocess_fomo.py` (`select` → `download`/`preprocess --fetch --delete-raw`) and `scripts/make_split.py` (`--num-test 70 --num-val 6 --num-train 500 --seed 0 --require-num-slices 176`). These are in Eshaan's local folder `~/personal/projects/mri-super-resolution/scripts/` and **not yet committed** to this repo.
    - 650 subjects were selected; 2 were skipped for shape (256, 320, 320): sub_502 and sub_2839.
    - sub_8112 ses_1 and sub_10915 ses_1 were verified bit-identical to the original bucket data.
    - **Overlap with the first submission:** the new test set contains 2 of the paper's 59 training subjects (and 7 of the 156 in the old bucket). The paper's 5 test subjects are in no new split. **Don't evaluate the old March checkpoints on the new test set**; retrain instead.
  - Why this size: 500/6/70 was Eshaan's choice, made after comparing split sizes in comparable papers:

    | Paper | Train / val / test |
    |---|---|
    | DCSRN and mDCSRN-GAN (HCP) | 780 / 111 (+111) / 111 |
    | DisC-Diff (IXI) | 500 / 6 / 70 |
    | PDM | 206/142, 600/200, 636/200 (train/test) |
    | Res-SRDiff | 121/21 and 268/66 (no val) |

    Brain-MRI SR test sets are typically 70–111 subjects. Not yet confirmed with Hendrik, who suggested about 30 test and 100–200 train subjects on the call (`../ASK_HENDRIK.md`).
  - **Cost per epoch vs. March:**
    - 2.5D: unchanged. An epoch is 400 optimizer steps whatever the dataset size: about 43 min on one L4 (0.8 s per batch in the 2026-10-24 Vertex smoke run; March: 42.5 min). One pass over the 88,000 training slices is about 3.4 epochs (25,600 target slices per epoch; estimate). The paper's 10 epochs were about 14.5 passes over its 17,600 slices.
    - Swin2SR and EDSR: 500 volumes instead of 100, so 5× per epoch.
    - 3D: 500 instead of 258, so about 1.9× (the 258 is from the code, NOT VERIFIED by a log).
- **E2 — Matched sampler.** Use the same EDM sampler (Euler or Heun; pick one and justify) with 20 steps for both 3D and 2.5D. Report the old 1-step 2.5D result for reference.
- **E3 — Small hyperparameter ablation.** Use a reduced budget (subset of train data, fewer steps) and the val split.
  - Sampling steps {1, 5, 20} for both models.
  - 2.5D context slices: {1 previous, both neighbours, 2 on each side}. Set with `num_steps_conditioning` (before) and `num_next_slices_conditioning` (after); see "Still open" item 7.
  - 3D patch size: {32³, 64³}.
  - EDM sigma distribution (P_mean, P_std): {(-1.2, 1.2), (-0.6, 1.0), (0.0, 1.0)}.
  - Attention: deepest level only vs. the two deepest levels (only if budget allows).
  - One factor at a time around the default. Table plus one figure.
- **E4 — Convergence.**
  - Add periodic validation (loss, PSNR, SSIM on a fixed val subset) to the 2.5D trainer (`src/trainer.py`), the 3D trainer (`src/train_3d.py`) and every baseline trainer.
  - Log to W&B and CSV.
  - Train each model until val PSNR plateaus, with the same budget in GPU-hours or optimizer steps.
  - Plot curves for all models. Model selection uses the best val checkpoint.
- **E5 — k-space degradation.** Implemented 2026-10-24: `src/data/degradation.py` (`degrade_kspace`) and `scripts/make_degraded_split.py`.
  - The function keeps the central half of k-space in-plane (truncation, so Gibbs ringing), optionally adds complex Gaussian noise (Rician in the magnitude), and shifts the LR grid by half an HR pixel so it lines up with block averaging. Without that shift the LR image is offset and bicubic loses about 3.5 dB for reasons unrelated to the degradation.
  - On one test volume: aligned k-space LR vs block-averaged LR 36.0 dB; bicubic from it 30.0 dB, against 29.7 dB from block averaging.
  - Test sets built from the 70-subject `evaluation` split: `gs://hendrik_mri_super_resolution/mri_denoiser/processed_data_mri_resub_kspace/evaluation/` (no noise) and `…_kspace_noise2/evaluation/` (noise std 2 in 0–255 units). Each has a `degradation.json`.
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
  - **Chosen: Res-SRDiff** (Safari et al., the paper's "Wu et al."); Xia et al.'s code is not public. Branch `resub/ressrdiff`: `src/ressrdiff_adapter.py`, `src/train_ressrdiff.py`, `scripts/run_training_job_ressrdiff.sh` (downloads the pinned commit at run time), `evaluate_sr.py --method ressrdiff`. Full run `mri-resub-ressrdiff-main-20261025-0435`: one A100, 12 h cap (~100k steps at batch 16), EMA weights validated every 5,500 steps (one pass) on 16 slices per val volume. It runs on an A100 while the others ran on L4s, so compare budgets in optimizer steps and samples seen, not hours.
- **E7 — Fair baselines.** EDSR and Swin2SR trained/fine-tuned on the same data and degradation. Bicubic stays as a reference. Both train from scratch with `src/train_swin2sr.py --arch {swin2sr,edsr}` ("Still open" item 3).
- **E8 — Statistics.** Implemented: `scripts/evaluate_sr.py` (per-slice CSV per method) and `scripts/compare_sr.py` (tables and tests).
  - **Final evaluation** `mri-resub-eval-main-20261024-2204` (A100, image 343c12b): the best-val checkpoints of all six models on the 70-volume test set, three degradations, 20 sampling steps, plus a 1/5-step sweep. Block-averaged results (per-volume mean PSNR / fg PSNR / SSIM / LPIPS-Alex): bicubic 33.66 / 30.91 / 0.9570 / 0.0865; EDSR 37.50 / 34.69 / 0.9769 / 0.0259; Swin2SR 37.11 / 34.43 / 0.9750 / 0.0295; Res-SRDiff 36.48 / 33.80 / 0.9719 / 0.0083; 2.5D 37.43 / 34.61 / 0.9768 / 0.0226; 3D 37.37 / 34.61 / 0.9771 / 0.0236. 2.5D both-neighbours, re-run with the edge fix (`mri-resub-eval-25dboth-edgefix-20261024-2204`, image 92cafdb): 37.70 / 34.88 / 0.9776 / 0.0224, the best model on block averaging (+0.20 dB over EDSR, +0.27 over one-sided 2.5D, +0.33 over 3D, all p < 1e-11). Its original 37.57 was measured with the bug. Tables with the fixed CSV: `results/final_eval_20261024-2204/{blockavg,kspace}/summary_edgefix.md`. The k-space test sets are still running.
  - What it means: the four best models are within 0.2 dB. 2.5D vs 3D: +0.06 dB PSNR (Holm p = 1e-7), fg PSNR tied (p = 0.72). The first submission's "3D best by 1.93 dB" does not hold. EDSR trained on the same data matches the diffusion models on PSNR/SSIM; the diffusion models win on LPIPS (0.023 vs 0.026).
  - Res-SRDiff's LPIPS is not independent evidence: it trains on 4 × MSE + 1 × LPIPS with `lpips.LPIPS(net='alex')`, the same network as our metric. Say so in the paper. A metric it was not trained on (LPIPS-VGG, DISTS) would need every method re-run, since predictions are not stored.
  - **Convergence (R1.2a).** Change in val PSNR over the last quarter of each main run: EDSR +0.00 dB, Res-SRDiff +0.06, 2.5D +0.08 (best at its last epoch), 2.5D both +0.12, Swin2SR +0.13 (only 6 epochs), 3D +1.01 (still rising). Curves: results/training_curves/ (not in the repo). Learning rates are constant, so the 3D climb is real. The four unconverged runs are being continued from their last weights with early stopping (`mri-resub-{3d,25d,25d-both,swin2sr}-cont-20261027-1750`, commit e1708de); their final evaluation must then be redone. Report both: equal budget (the main runs) and to convergence (the continued runs). The 2.5D trainer's logged val_loss_denoising column is wrong for every run before ce421cc: make_segment handed each segment the episode's own info dict, so DatasetTraverser's chunks all carried the last chunk's full_res and the loss scored slice 0 against slice 175 (3.6 rising to 6.2; the real value is 0.053 at epoch 1 and 0.013 at epoch 30, tracking the training loss). Fixed in ce421cc (a dict copy). Val PSNR, checkpoint selection, training and all test results were unaffected. For the R1.2a curve, use the recomputed column (scripts/recompute_val_loss.py over the saved epoch checkpoints, job context/eval_resub/recompute_val_loss_job.sh; done for both main 2.5D runs: 0.052 → 0.013 and 0.032 → 0.012 over 30 epochs, flat from epoch ~20).
  - **k-space test set (E5a), 2026-10-28:** per-volume mean PSNR / fg PSNR / SSIM / LPIPS: bicubic 33.93 / 31.22 / 0.9582 / 0.0767; EDSR 35.01 / 32.24 / 0.9619 / 0.0410; Swin2SR 34.92 / 32.26 / 0.9610 / 0.0448; Res-SRDiff 34.20 / 31.50 / 0.9548 / 0.0187; 2.5D 34.80 / 32.00 / 0.9614 / 0.0353; 2.5D both 34.98 / 32.17 / 0.9621 / 0.0361; 3D 34.78 / 32.04 / 0.9604 / 0.0390. The learned models gain ~1 dB over bicubic here against ~3.8 dB on block averaging. EDSR and 2.5D-both are within 0.04 dB (p = 2e-5 in EDSR's favour); 2.5D vs 3D is not significant on PSNR (p = 0.09); the diffusion models keep the LPIPS lead over EDSR and Swin2SR. 
  - **Noisy k-space set (E5a, Rician noise std 2/255), 2026-10-28:** PSNR / fg / SSIM / LPIPS: bicubic 32.63 / 30.93 / 0.9445 / 0.1031; EDSR 33.26 / 31.67 / 0.9415 / 0.0805; Swin2SR 32.92 / 31.51 / 0.9408 / 0.0830; Res-SRDiff 33.00 / 30.76 / 0.9349 / 0.0577; 2.5D 33.30 / 31.46 / 0.9440 / 0.0615; 2.5D both 33.54 / 31.70 / 0.9457 / 0.0612; 3D 33.06 / 31.44 / 0.9419 / 0.0705. With noise the diffusion models lead on every metric (2.5D-both over EDSR: +0.28 dB, p 4e-13); bicubic's SSIM beats every learned model except 2.5D-both, because the models were trained on noise-free block averaging. Say in the paper that none of the models saw noise in training. Tables: `results/final_eval_20261024-2204/kspace_noise2/summary_edgefix.md`.
  - **Sampling-step sweep (E3, block-averaged set, PSNR):** 2.5D 36.85 / 37.40 / 37.43 and 3D 32.98 / 37.39 / 37.37 at 1 / 5 / 20 Euler steps; inference per volume on an A100 2.5D 6 / 28 / 111 s, 3D 18 / 88 / 350 s. Five steps match twenty for both; one step costs 2.5D 0.6 dB and 3D 4.4 dB. The first submission's one-step 2.5D vs 20-step 3D pairing would put 3D ahead by 0.5 dB on this data; at matched steps 2.5D is ahead by 0.06. Table: `results/final_eval_20261024-2204/steps/summary.md`.
  - **Continuation outcomes (2026-10-28):** 2.5D stopped after 5 epochs with no new best (epoch 30 stands, converged). 2.5D both: new best at continuation epoch 31 (val 37.90 vs 37.84 at the start, measured with the edge fix), then 5 epochs without improvement; its epoch-31 checkpoint, evaluated on the three test sets (`mri-resub-eval-converged-25dboth-20261027-1750`, script `context/eval_resub/eval_converged_job.sh`, SPECS via EXTRA_ENV), scores 37.69 / 34.87 / 33.45 dB (blockavg / k-space / noisy k-space) against 37.70 / 34.98 / 33.54 for epoch 27: the validation gain did not transfer (−0.01, −0.11, −0.09 dB, all p < 2e-5). Report epoch 31 in the to-convergence table (best-val protocol) and epoch 27 in the equal-budget table; never pick by test score. 3D is still running (best so far epoch 33, 37.15, +0.11 over the start, noisy). Swin2SR is still running on an L4 (epoch 7: 37.43, +0.03). Evaluate the 3D and Swin2SR continuation checkpoints with the same script when they stop. Note: the fork resubmitted a second, identical Swin2SR continuation (`mri-resub-swin2sr-cont-20261028-0040`); only `…-cont-l4-20261027-1750` (W&B run bhy92f8j) is the one to keep.
  - Per-subject metrics, mean ± SD, bootstrap 95% CI over subjects.
  - Paired Wilcoxon signed-rank between methods (subject level, with Holm correction).
  - Per-slice plots with bands.
- **E9 — External validation.** IXI T1 with the same preprocessing and zero-shot inference.
- **E10 — Fidelity / downstream.** SynthSeg on GT vs. SR volumes, Dice per structure; failure cases; difference maps.

## Phases

0. **Recovery & reproducibility.**
   - Recover the MRI preprocessing script, the Swin2SR training code and the subject lists from: the Vertex training images in Artifact Registry (`omgrab-dev`), Vertex custom-job specs and Cloud Logging, the run directories, the bucket metadata (`episode_mapping.json`, `info.pt*`) and any local copies.
   - Commit what we recover.
   - Done 2026-10-24: the published checkpoints were re-evaluated on the rebuilt paper test set, and every number in the paper's Tables 1 and 2 is accounted for ("How the first submission's numbers were produced"). Do not evaluate the March checkpoints on the new test set: 2 of its subjects were in the March training data.
1. **Data.** E1 splits, plus the k-space test set (E5a) and IXI (E9).
2. **Tooling.** Validation logging (E4), a single eval harness with statistics (E8), compute profiling, and the `degrade_kspace` function.
3. **Main training.** 3D, 2.5D, EDSR, Swin2SR and the new diffusion model on the new split.
4. **Ablations.** E3, and E5b.
5. **Analysis.** E10, figures, tables.
6. **Writing.** Update `paper.tex` (`\documentclass{ieeeaccess}`, the file to resubmit to IEEE Access); `preprint_arxiv.tex` is the IEEEtran arXiv version, kept in sync if wanted. Then prepare the highlighted-changes PDF, and write the response-to-reviewers document (one entry per comment, matching the triage table).

## Conventions

- Training on Vertex AI uses `scripts/launch_vertex_training.py`. Write new runs to `gs://omgrab-dev-diamond-vertex-artifacts/diamond/runs/mri-resub-<descriptive-name>/`. Other people's runs share `diamond/runs/`, so every new run name starts with `mri-resub-`.
- Never delete or overwrite anything in `omgrab-dev`: not the original paper runs, not the processed data, not other people's runs. Write new processed data under a new prefix (the resubmission data is in `processed_data_mri_resub/`).
- Ask Eshaan before launching any Vertex job.
- Never push `diamond-train:latest` (the launcher's `--push-latest` does). Build resubmission images as `us-central1-docker.pkg.dev/omgrab-dev/diamond-training/mri-resub-train:<git sha>` from a clean `git archive` of that commit, and pass `--image-uri`.
- Until a W&B account exists, launch with `--wandb-secret ""`, and add `wandb.mode=offline` (2.5D) or `--wandb-project=` (3D, baselines). Otherwise the 2.5D trainer retries `wandb.init` forever (`try_until_no_except`). The convergence data is in each run's `val_metrics.csv`.
- Do not commit data, checkpoints or `.nii` files (already covered by `.gitignore`).
- Keep the triage table above up to date as decisions change.
