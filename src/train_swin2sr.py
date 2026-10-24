"""Standalone training script for the 2D SR baselines on MRI slices (from scratch).

Trains Swin2SR-lightweight (the caidas/swin2SR-lightweight-x2-64 architecture)
or EDSR-base (the eugenesiow/edsr-base architecture) from random initialization
on 2D MRI slice pairs (LR -> HR, scale factor 2) with an L1 loss.

Validation PSNR/SSIM on evenly spaced slices of every validation volume go to
W&B and ``val_metrics.csv``; the best epoch is saved as ``<arch>_best_val.pt``.

Usage:
    python src/train_swin2sr.py --arch swin2sr \
        --lr-train <lr_dir>/train --hr-train <hr_dir>/train \
        --lr-test <lr_dir>/test --hr-test <hr_dir>/test \
        --epochs 20 --batch-size 16 --lr 2e-4

On Vertex, EDSR goes through the same job script:
    launch_vertex_training.py --mode swin2sr --train-swin2sr-args "--arch edsr"
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import Dataset, MRIHdf5Dataset
from metrics import psnr as compute_psnr, ssim as compute_ssim, lpips_distance as compute_lpips

try:
    import wandb
except ImportError:
    wandb = None


class MRISlicePairDataset(TorchDataset):
    """Flat dataset of (LR, HR) 2D MRI slice pairs for supervised SR training.

    LR slices come from the DIAMOND-style episode .pt files; HR slices from
    the companion HDF5 volumes. Both are stored as uint8 in two contiguous
    tensors and returned in [0, 1] as 3-channel tensors (grayscale repeated).
    """

    def __init__(
        self, lr_dir: Path, hr_dir: Path, split: str = "train", max_slices: int = 0,
    ) -> None:
        super().__init__()
        self._hr_dir = Path(hr_dir)

        hr_dataset = MRIHdf5Dataset(hr_dir)
        lr_dataset = Dataset(lr_dir, dataset_full_res=hr_dataset, name=split)
        lr_dataset.load_from_default_path()

        capacity = lr_dataset.num_steps if max_slices <= 0 else min(max_slices, lr_dataset.num_steps)
        lr_all = hr_all = None
        n_pairs = 0
        self.volume_ranges: list[tuple[int, int]] = []  # (first pair index, number of pairs) per volume
        for ep_id in range(lr_dataset.num_episodes):
            ep = lr_dataset.load_episode(ep_id)
            obs_lr = ep.obs.float()

            file_id = ep.info.get("original_file_id", f"episode_{ep_id}")
            hr_file_id = file_id
            if hr_file_id.startswith(f"{split}/"):
                hr_file_id = hr_file_id[len(f"{split}/"):]
            hr_path = self._hr_dir / hr_file_id
            if not hr_path.exists():
                print(f"  Skipping {file_id}: HR file not found at {hr_path}", flush=True)
                continue

            with h5py.File(hr_path, "r") as f:
                frame_keys = sorted(
                    [k for k in f.keys() if k.endswith("_x")],
                    key=lambda k: int(k.split("_")[1]),
                )
                hr_slices = np.stack([np.asarray(f[k]) for k in frame_keys])

            n = min(obs_lr.size(0), len(hr_slices), capacity - n_pairs)
            if lr_all is None:
                lr_all = torch.empty((capacity, *obs_lr.shape[1:]), dtype=torch.uint8)
                hr_all = torch.empty((capacity, 1, *hr_slices.shape[1:]), dtype=torch.uint8)
            # Episode obs are uint8 / 255 * 2 - 1; rounding recovers the stored uint8 exactly.
            lr_all[n_pairs : n_pairs + n] = obs_lr[:n].add(1).div(2).mul(255).round().to(torch.uint8)
            hr_all[n_pairs : n_pairs + n, 0] = torch.from_numpy(hr_slices[:n])
            self.volume_ranges.append((n_pairs, n))
            n_pairs += n
            if n_pairs >= capacity:
                break

        self._lr = lr_all[:n_pairs] if lr_all is not None else torch.empty(0, 1, 1, 1, dtype=torch.uint8)
        self._hr = hr_all[:n_pairs] if hr_all is not None else torch.empty(0, 1, 1, 1, dtype=torch.uint8)
        print(f"MRISlicePairDataset [{split}]: {n_pairs} slice pairs from {len(self.volume_ranges)} volumes", flush=True)

    def __len__(self) -> int:
        return self._lr.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        lr = self._lr[idx].float().div(255)
        hr = self._hr[idx].float().div(255)
        lr_3ch = lr.repeat(3, 1, 1)
        hr_3ch = hr.repeat(3, 1, 1)
        return {"lr": lr_3ch, "hr": hr_3ch, "lr_gray": lr, "hr_gray": hr}


def build_swin2sr(device: torch.device):
    """Build a Swin2SR-lightweight model from scratch (random weights).

    Uses the same architecture as caidas/swin2SR-lightweight-x2-64 but with
    random initialization so the model learns entirely from MRI data.
    """
    from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

    config = Swin2SRConfig.from_pretrained("caidas/swin2SR-lightweight-x2-64")
    model = Swin2SRForImageSuperResolution(config)
    model.train()
    return model.to(device)


def build_edsr(device: torch.device):
    """Build an EDSR-base model from scratch (random weights).

    Same architecture as eugenesiow/edsr-base: 16 residual blocks, 64 features, x2.
    """
    from super_image import EdsrConfig, EdsrModel

    model = EdsrModel(EdsrConfig(scale=2, n_resblocks=16, n_feats=64, n_colors=3, res_scale=1))
    model.train()
    return model.to(device)


BUILDERS = {"swin2sr": build_swin2sr, "edsr": build_edsr}


def predict(model: torch.nn.Module, lr_3ch: torch.Tensor) -> torch.Tensor:
    out = model(lr_3ch)
    return out.reconstruction if hasattr(out, "reconstruction") else out


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: MRISlicePairDataset,
    device: torch.device,
    slices_per_volume: int = 16,
    with_lpips: bool = False,
) -> dict[str, float]:
    """PSNR/SSIM (and optionally LPIPS) on evenly spaced slices of every validation volume."""
    model.eval()
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    for first, n in dataset.volume_ranges:
        for i in np.unique(np.linspace(0, n - 1, min(slices_per_volume, n)).round().astype(int)):
            sample = dataset[first + int(i)]
            lr_3ch = sample["lr"].unsqueeze(0).to(device)
            hr_gray = sample["hr_gray"].unsqueeze(0).to(device)

            pred = predict(model, lr_3ch).clamp(0, 1)
            pred_gray = pred.mean(dim=1, keepdim=True)

            pred_m11 = pred_gray.mul(2).sub(1)
            hr_m11 = hr_gray.mul(2).sub(1)

            pred_m11 = pred_m11[:, :, : hr_m11.size(2), : hr_m11.size(3)]

            psnr_vals.append(compute_psnr(pred_m11, hr_m11))
            ssim_vals.append(compute_ssim(pred_m11, hr_m11))
            if with_lpips:
                lpips_vals.append(compute_lpips(pred_m11, hr_m11))

    model.train()
    metrics = {
        "eval/psnr": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
        "eval/ssim": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
        "eval/num_slices": len(psnr_vals),
    }
    if with_lpips:
        metrics["eval/lpips"] = float(np.mean(lpips_vals)) if lpips_vals else 0.0
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2D SR baseline (Swin2SR or EDSR) from scratch on MRI slices.")
    parser.add_argument("--arch", choices=sorted(BUILDERS), default="swin2sr")
    parser.add_argument("--lr-train", type=Path, required=True, help="LR train directory")
    parser.add_argument("--hr-train", type=Path, required=True, help="HR train directory (.hdf5)")
    parser.add_argument("--lr-test", type=Path, default=None, help="LR validation directory (never the held-out test set)")
    parser.add_argument("--hr-test", type=Path, default=None, help="HR validation directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-acc-steps", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--eval-slices-per-volume", type=int, default=16)
    parser.add_argument("--eval-lpips", action="store_true", help="Also compute LPIPS during validation")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints_swin2sr"))
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-slices", type=int, default=0,
                        help="Cap training slices (0 = use all)")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print step-level loss every N steps")
    return parser.parse_args()


def main() -> None:
    import os
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}", flush=True)

    if args.wandb_project and wandb is not None:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    print(f"Building {args.arch} from scratch (random init) ...", flush=True)
    model = BUILDERS[args.arch](device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    print("Building training dataset ...", flush=True)
    train_dataset = MRISlicePairDataset(
        args.lr_train, args.hr_train, split="train", max_slices=args.max_train_slices,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataset = MRISlicePairDataset(args.lr_test, args.hr_test, split="test") if args.lr_test and args.hr_test else None

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val_psnr = float("-inf")
    path_csv = args.save_dir / "val_metrics.csv"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        total_steps = len(train_loader)
        for step, batch in enumerate(train_loader):
            lr_3ch = batch["lr"].to(device)
            hr_3ch = batch["hr"].to(device)

            pred = predict(model, lr_3ch)
            pred = pred[:, :, : hr_3ch.size(2), : hr_3ch.size(3)]

            loss = F.l1_loss(pred, hr_3ch)
            loss = loss / args.grad_acc_steps
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_acc_steps
            n_batches += 1

            if (step + 1) % args.log_every == 0:
                step_loss = epoch_loss / n_batches
                elapsed = time.time() - t0
                print(
                    f"  [epoch {epoch}] step {step + 1}/{total_steps}  "
                    f"loss={step_loss:.6f}  elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if (step + 1) % args.grad_acc_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.6f}  time={elapsed:.1f}s", flush=True)

        log_dict: dict[str, float] = {"epoch": epoch, "train/loss": avg_loss}

        if val_dataset is not None and epoch % args.eval_every == 0:
            eval_metrics = evaluate(model, val_dataset, device, args.eval_slices_per_volume, args.eval_lpips)
            log_dict.update(eval_metrics)
            lpips_msg = f"  LPIPS={eval_metrics['eval/lpips']:.4f}" if "eval/lpips" in eval_metrics else ""
            print(f"  val: PSNR={eval_metrics['eval/psnr']:.2f}  SSIM={eval_metrics['eval/ssim']:.4f}{lpips_msg}", flush=True)

            is_best = eval_metrics["eval/psnr"] > best_val_psnr
            if is_best:
                best_val_psnr = eval_metrics["eval/psnr"]
                best_path = args.save_dir / f"{args.arch}_best_val.pt"
                torch.save(model.state_dict(), best_path)
                print(f"  New best val PSNR; saved {best_path}", flush=True)

            row = {"epoch": epoch, "train_loss": avg_loss, "val_psnr": eval_metrics["eval/psnr"],
                   "val_ssim": eval_metrics["eval/ssim"], "is_best": int(is_best)}
            write_header = not path_csv.is_file()
            with path_csv.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

        if args.wandb_project and wandb is not None:
            wandb.log(log_dict)

        if epoch % args.save_every == 0:
            ckpt_path = args.save_dir / f"{args.arch}_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved {ckpt_path}", flush=True)

    final_path = args.save_dir / f"{args.arch}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model: {final_path}", flush=True)


if __name__ == "__main__":
    main()
