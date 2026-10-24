"""Standalone training script for the 3D volumetric MRI super-resolution model.

Usage:
    python train_3d.py --config-name trainer_mri_3d
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from data.dataset_3d import MRIPatchDataset3d, MRIVolumeDataset3d
from models.diffusion.denoiser_3d import Denoiser3d, Denoiser3dConfig
from models.diffusion.inner_model_3d import InnerModel3dConfig
from models.diffusion.denoiser import SigmaDistributionConfig
from models.diffusion.patchwise_3d import patchwise_sample
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
    parser.add_argument("--lr-test", type=Path, default=None, help="LR validation directory (used for checkpoint selection, never the held-out test set)")
    parser.add_argument("--hr-test", type=Path, default=None, help="HR validation directory")
    parser.add_argument("--eval-every", type=int, default=1, help="Validation loss every N epochs")
    parser.add_argument("--eval-psnr-every", type=int, default=None, help="Validation PSNR/SSIM every N epochs (default: --save-every)")
    parser.add_argument("--eval-max-volumes", type=int, default=4)
    parser.add_argument("--eval-num-steps", type=int, default=20)
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
def validation_loss(model: Denoiser3d, val_patches: MRIPatchDataset3d, batch_size: int, device: torch.device, seed: int = 0) -> float:
    """Mean denoising loss on validation patches, with fixed patches and noise levels every epoch."""
    np_state = np.random.get_state()
    np.random.seed(seed)
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        losses = []
        for batch in DataLoader(val_patches, batch_size=batch_size, shuffle=False, num_workers=0):
            loss, _ = model(batch["lr"].to(device), batch["hr"].to(device))
            losses.append(loss.item())
    np.random.set_state(np_state)
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate_3d(
    model: Denoiser3d,
    lr_test: Path,
    hr_test: Path,
    upsampling_factor: int,
    device: torch.device,
    patch_size: int,
    num_steps: int = 20,
    max_volumes: int = 4,
) -> dict:
    """Sliding-window sampling on validation volumes; PSNR/SSIM are 2-D per slice, averaged per volume."""
    model.eval()
    test_ds = MRIVolumeDataset3d(lr_test, hr_test, upsampling_factor)
    psnr_vals, ssim_vals = [], []

    for i in range(min(len(test_ds), max_volumes)):
        sample = test_ds[i]
        lr_vol = sample["lr"].unsqueeze(0)
        hr_vol = sample["hr"].unsqueeze(0)

        pred_hr = patchwise_sample(
            model, lr_vol, device, patch_size, patch_size // 4, upsampling_factor, num_steps, verbose=False,
        )
        d = min(pred_hr.size(2), hr_vol.size(2))
        psnr_vals.append(sum(compute_psnr(pred_hr[:, :, si], hr_vol[:, :, si]) for si in range(d)) / d)
        ssim_vals.append(sum(compute_ssim(pred_hr[:, :, si], hr_vol[:, :, si]) for si in range(d)) / d)

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

    val_patches = None
    if args.lr_test and args.hr_test:
        val_patches = MRIPatchDataset3d(
            lr_dir=args.lr_test,
            hr_dir=args.hr_test,
            patch_size=args.patch_size,
            upsampling_factor=args.upsampling_factor,
            patches_per_volume=args.patches_per_volume,
        )
        print(f"Val: {len(val_patches)} patches from {val_patches.num_volumes} volumes")
    eval_psnr_every = args.eval_psnr_every or args.save_every
    best_val_psnr = float("-inf")

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

        if val_patches is not None and epoch % args.eval_every == 0:
            val_loss = validation_loss(model, val_patches, args.batch_size, device)
            log_dict["val/loss"] = val_loss
            print(f"  val: loss={val_loss:.4f}")

        if val_patches is not None and epoch % eval_psnr_every == 0:
            eval_metrics = evaluate_3d(
                model, args.lr_test, args.hr_test, args.upsampling_factor, device,
                args.patch_size, num_steps=args.eval_num_steps, max_volumes=args.eval_max_volumes,
            )
            log_dict.update(eval_metrics)
            print(f"  val: PSNR={eval_metrics.get('eval/psnr', 0):.2f}  SSIM={eval_metrics.get('eval/ssim', 0):.4f}")
            if eval_metrics["eval/psnr"] > best_val_psnr:
                best_val_psnr = eval_metrics["eval/psnr"]
                log_dict["is_best"] = 1
                torch.save(model.state_dict(), args.save_dir / "model_best_val.pt")
                print(f"  New best val PSNR; saved {args.save_dir / 'model_best_val.pt'}")

        if args.wandb_project and wandb is not None:
            wandb.log(log_dict)

        if val_patches is not None:
            row = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": log_dict.get("val/loss", ""),
                "val_psnr": log_dict.get("eval/psnr", ""),
                "val_ssim": log_dict.get("eval/ssim", ""),
                "is_best": log_dict.get("is_best", 0),
            }
            path_csv = args.save_dir / "val_metrics.csv"
            write_header = not path_csv.is_file()
            with path_csv.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

        if epoch % args.save_every == 0:
            ckpt_path = args.save_dir / f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved {ckpt_path}")

    final_path = args.save_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
