"""Standalone training script for the 3D volumetric MRI super-resolution model.

Usage:
    python train_3d.py --config-name trainer_mri_3d
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from data.dataset_3d import MRIPatchDataset3d, MRIVolumeDataset3d
from models.diffusion.denoiser_3d import Denoiser3d, Denoiser3dConfig
from models.diffusion.inner_model_3d import InnerModel3dConfig
from models.diffusion.denoiser import SigmaDistributionConfig
from metrics import psnr as compute_psnr, ssim as compute_ssim

try:
    import wandb
except ImportError:
    wandb = None


def build_model(cfg: dict) -> Denoiser3d:
    inner_cfg = InnerModel3dConfig(**cfg["inner_model"])
    denoiser_cfg = Denoiser3dConfig(
        inner_model=inner_cfg,
        sigma_data=cfg["sigma_data"],
        sigma_offset_noise=cfg["sigma_offset_noise"],
        upsampling_factor=cfg["upsampling_factor"],
    )
    return Denoiser3d(denoiser_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D MRI super-resolution model.")
    parser.add_argument("--lr-train", type=Path, required=True, help="LR train directory (.pt files)")
    parser.add_argument("--hr-train", type=Path, required=True, help="HR train directory (.hdf5 files)")
    parser.add_argument("--lr-test", type=Path, default=None, help="LR test directory")
    parser.add_argument("--hr-test", type=Path, default=None, help="HR test directory")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--upsampling-factor", type=int, default=2)
    parser.add_argument("--patches-per-volume", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-acc-steps", type=int, default=4)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 2, 2, 2])
    parser.add_argument("--attn-depths", type=int, nargs="+", default=[0, 0, 0, 1])
    parser.add_argument("--cond-channels", type=int, default=1024)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints_3d"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug-step-logs", action="store_true", help="Print detailed logs every N train steps.")
    parser.add_argument("--debug-log-every", type=int, default=1, help="Log every N train steps when debug enabled.")
    return parser.parse_args()


@torch.no_grad()
def evaluate_3d(
    model: Denoiser3d,
    lr_test: Path,
    hr_test: Path,
    upsampling_factor: int,
    device: torch.device,
    num_steps: int = 20,
    max_volumes: int = 4,
) -> dict:
    """Run evaluation on test volumes and compute PSNR/SSIM."""
    from data.dataset_3d import MRIVolumeDataset3d

    model.eval()
    test_ds = MRIVolumeDataset3d(lr_test, hr_test, upsampling_factor)
    psnr_vals, ssim_vals = [], []

    for i in range(min(len(test_ds), max_volumes)):
        sample = test_ds[i]
        lr_vol = sample["lr"].unsqueeze(0).to(device)
        hr_vol = sample["hr"].unsqueeze(0).to(device)

        pred_hr = model.sample(lr_vol, num_steps=num_steps)

        psnr_vals.append(compute_psnr(pred_hr, hr_vol))
        ssim_vals.append(compute_ssim(pred_hr, hr_vol))

    model.train()
    return {
        "eval/psnr": sum(psnr_vals) / len(psnr_vals) if psnr_vals else 0,
        "eval/ssim": sum(ssim_vals) / len(ssim_vals) if ssim_vals else 0,
    }


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if args.wandb_project and wandb is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    model_cfg = {
        "inner_model": {
            "img_channels": 1,
            "cond_channels": args.cond_channels,
            "depths": args.depths,
            "channels": args.channels,
            "attn_depths": args.attn_depths,
        },
        "sigma_data": 0.5,
        "sigma_offset_noise": 0.1,
        "upsampling_factor": args.upsampling_factor,
    }
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    sigma_cfg = SigmaDistributionConfig(loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=20)
    model.setup_training(sigma_cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    train_dataset = MRIPatchDataset3d(
        lr_dir=args.lr_train,
        hr_dir=args.hr_train,
        patch_size=args.patch_size,
        upsampling_factor=args.upsampling_factor,
        patches_per_volume=args.patches_per_volume,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_dataset)} patches from {train_dataset.num_volumes} volumes")

    args.save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            lr_patch = batch["lr"].to(device)
            hr_patch = batch["hr"].to(device)

            loss, metrics = model(lr_patch, hr_patch)
            loss = loss / args.grad_acc_steps
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += metrics["loss_3d_upsampler"]
            n_batches += 1

            if args.debug_step_logs and ((step + 1) % max(args.debug_log_every, 1) == 0):
                print(
                    f"[debug][train3d] epoch={epoch} step={step + 1}/{len(train_loader)} "
                    f"loss={metrics['loss_3d_upsampler']:.6f} "
                    f"lr_shape={tuple(lr_patch.shape)} hr_shape={tuple(hr_patch.shape)} "
                    f"device={device}"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s")

        log_dict = {"epoch": epoch, "train/loss": avg_loss}

        if args.lr_test and args.hr_test and epoch % args.save_every == 0:
            eval_metrics = evaluate_3d(model, args.lr_test, args.hr_test, args.upsampling_factor, device)
            log_dict.update(eval_metrics)
            print(f"  eval: PSNR={eval_metrics.get('eval/psnr', 0):.2f}  SSIM={eval_metrics.get('eval/ssim', 0):.4f}")

        if args.wandb_project and wandb is not None:
            wandb.log(log_dict)

        if epoch % args.save_every == 0:
            ckpt_path = args.save_dir / f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved {ckpt_path}")

    final_path = args.save_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
