# MRI Super-Resolution via Elucidated Diffusion Models

Comparative analysis of **3D convolutional** and **2.5D slice-conditioned** U-Net architectures for brain MRI super-resolution using the Elucidated Diffusion Model (EDM) framework.

**Pretrained Weights:** [HuggingFace](https://huggingface.co/Chichonnade/Comparative-Analysis-3D-2.5D-MRI-Super-Resolution-EDM)
**Dataset:** [FOMO60K / NKI cohort](https://huggingface.co/datasets/FOMO-MRI/FOMO60K)

## Results

| Method | PSNR (dB) | SSIM | Params |
|---|---|---|---|
| Bicubic interpolation | 31.01 | 0.952 | -- |
| Trilinear interpolation | 33.85 | 0.988 | -- |
| 2.5D EDM (ours, 10 ep) | 33.68 | 0.967 | 51.1M |
| **3D EDM (ours, 10 ep)** | **37.77** | **0.996** | 50.7M |

Evaluated on 5 held-out NKI subjects (6 volumes, 993 sagittal slices) with 2x super-resolution.

## Architecture

Both models use the EDM framework (Karras et al. 2022) adapted from DIAMOND:

- **3D EDM**: Full 3D convolutional U-Net with channels [32, 64, 128, 256], 3D self-attention at the deepest level, patch-based training (32^3) and sliding-window inference with overlap blending. 20-step Euler sampling.
- **2.5D EDM**: 2D U-Net with channels [64, 64, 128, 256], conditions on 1 adjacent slice for inter-slice context. Single-step Heun sampling (0.09s/slice).

## Quick Start

### Install

```bash
pip install -e .
```

### Preprocess data

```bash
python src/raw_data/scripts/main.py src/processed_data_mri src/raw_data/FOMO60K/PT015_NKI \
  --mode 2d --axis sagittal --scale-factor 2
```

### Train 2.5D model (local)

```bash
python src/main.py --config-name trainer_mri \
  common.devices=cpu \
  training.num_final_epochs=10 \
  upsampler.training.batch_size=4 \
  upsampler.training.grad_acc_steps=8
```

### Train on Vertex AI (cloud)

```bash
python scripts/launch_vertex_training.py \
  --preprocessed-data-uri gs://your-bucket/processed_data_mri \
  --project your-project --region us-central1 \
  --service-account your-sa@project.iam.gserviceaccount.com \
  --image-uri your-image-uri \
  --artifact-bucket-uri gs://your-bucket/diamond \
  --epochs 10 --push-latest --yes
```

For 3D training, add `--mode 3d`.

### Run inference

```bash
# 2.5D inference from a cloud run
python scripts/run_inference_from_bucket.py run-YYYYMMDD-HHMMSS --evaluate

# 3D inference (local)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_inference_3d.py \
  --checkpoint path/to/model_final.pt --num-samples 6 --num-steps 20
```

### Evaluate

```bash
python scripts/evaluate_25d.py \
  --checkpoint path/to/agent_epoch_00010.pt --num-steps 1
```

## Project Structure

```
config/              Hydra configs (trainer_mri.yaml, agent/mri.yaml, env/mri.yaml)
scripts/             Training launchers, inference, evaluation
src/
  main.py            2.5D training entry point (Hydra)
  train_3d.py        3D training entry point (argparse)
  trainer.py         Core training loop
  models/
    diffusion/       EDM denoiser (2D + 3D variants)
    blocks.py        2D UNet building blocks
    blocks3d.py      3D UNet building blocks
  data/
    dataset.py       2D slice dataset
    dataset_3d.py    3D patch/volume dataset
  metrics.py         PSNR and SSIM implementations
```

## Authors

- **Hendrik Chiche** -- University of California, Berkeley
- **Ludovic Corcos**
- **Logan Rouge** -- ISIMA, Clermont Auvergne INP

In collaboration with GENCI.

## License

This project adapts the [DIAMOND](https://github.com/eloialonso/diamond) codebase for MRI super-resolution.
