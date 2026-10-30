"""Train Res-SRDiff (the MRI-specific diffusion comparator) from scratch on our slice pairs.

The network, diffusion, loss (MSE + LPIPS), EMA and optimizer are Res-SRDiff's
own (see src/ressrdiff_adapter.py); the data, validation and checkpoint
selection match our other trainers: validation PSNR/SSIM on evenly spaced
slices of every validation volume, sampled with the EMA weights,
``ressrdiff_best_val.pt`` (EMA weights) and ``val_metrics.csv``.

Usage:
    python src/train_ressrdiff.py --code-dir <Res-SRDiff clone> \
        --lr-train <lr>/train --hr-train <hr>/train --lr-test <lr>/test --hr-test <hr>/test \
        --save-dir checkpoints_ressrdiff --max-hours 22
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import psnr as compute_psnr, ssim as compute_ssim  # noqa: E402
from ressrdiff_adapter import build_model, load_ema_into_model, lq_from_lr, make_flags, sample  # noqa: E402
from train_swin2sr import MRISlicePairDataset  # noqa: E402


class PairsForResSRDiff(TorchDataset):
    """{'hq', 'lq'} in [-1, 1] at HR size, from the uint8 slice pairs of train_swin2sr.MRISlicePairDataset."""

    def __init__(self, pairs: MRISlicePairDataset) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.pairs[idx]
        lr = item["lr_gray"].mul(2).sub(1)
        return {"hq": item["hr_gray"].mul(2).sub(1), "lq": lq_from_lr(lr.unsqueeze(0))[0]}


@torch.no_grad()
def validate(rmodel, val: MRISlicePairDataset, slices_per_volume: int, seed: int) -> dict[str, float]:
    load_ema_into_model(rmodel)
    torch.manual_seed(seed)
    psnrs, ssims = [], []
    for first, n in val.volume_ranges:
        for i in np.unique(np.linspace(0, n - 1, min(slices_per_volume, n)).round().astype(int)):
            item = val[first + int(i)]
            lq = lq_from_lr(item["lr_gray"].mul(2).sub(1).unsqueeze(0))
            pred = sample(rmodel, lq).cpu()
            hr = item["hr_gray"].mul(2).sub(1).unsqueeze(0)
            psnrs.append(compute_psnr(pred, hr))
            ssims.append(compute_ssim(pred, hr))
    rmodel.model.train()
    return {"val_psnr": float(np.mean(psnrs)), "val_ssim": float(np.mean(ssims))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code-dir", type=Path, required=True, help="Clone of github.com/mosaf/Res-SRDiff")
    p.add_argument("--lr-train", type=Path, required=True)
    p.add_argument("--hr-train", type=Path, required=True)
    p.add_argument("--lr-test", type=Path, default=None, help="LR validation directory")
    p.add_argument("--hr-test", type=Path, default=None, help="HR validation directory")
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints_ressrdiff"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N optimizer steps (0 = no limit)")
    p.add_argument("--max-hours", type=float, default=0, help="Stop after N hours of training (0 = no limit)")
    p.add_argument("--eval-every-steps", type=int, default=5000)
    p.add_argument("--eval-slices-per-volume", type=int, default=16)
    p.add_argument("--max-train-slices", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    flags = make_flags(batch_size=args.batch_size, use_amp=args.use_amp, out_dir=str(args.save_dir / "ressrdiff_internal"))
    rmodel = build_model(args.code_dir, flags, device)
    n_params = sum(p.numel() for p in rmodel.model.parameters())
    print(f"Res-SRDiff parameters: {n_params:,}", flush=True)

    train = PairsForResSRDiff(MRISlicePairDataset(args.lr_train, args.hr_train, split="train", max_slices=args.max_train_slices))
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", drop_last=True)
    val = MRISlicePairDataset(args.lr_test, args.hr_test, split="test") if args.lr_test and args.hr_test else None

    path_csv = args.save_dir / "val_metrics.csv"
    best_psnr = float("-inf")
    step, t0 = 0, time.time()
    done = False

    def run_validation(epoch: int) -> None:
        nonlocal best_psnr
        metrics = validate(rmodel, val, args.eval_slices_per_volume, args.seed)
        is_best = metrics["val_psnr"] > best_psnr
        if is_best:
            best_psnr = metrics["val_psnr"]
            torch.save(rmodel.ema_state, args.save_dir / "ressrdiff_best_val.pt")
        row = {"step": step, "epoch": epoch, "hours": round((time.time() - t0) / 3600, 3), **metrics, "is_best": int(is_best)}
        write_header = not path_csv.is_file()
        with path_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"  val step {step}: PSNR={metrics['val_psnr']:.2f}  SSIM={metrics['val_ssim']:.4f}{'  (best)' if is_best else ''}", flush=True)

    rmodel.model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            hq = batch["hq"].to(device)
            lq = batch["lq"].to(device)
            t = torch.randint(0, flags.steps, size=(hq.size(0),), device=device)
            noise = torch.randn_like(hq)
            compute_losses = rmodel.base_diffusion.training_losses(
                model=rmodel.model, x_start=hq, y=lq, t=t, first_stage_model=None,
                model_kwargs={"lq": lq}, noise=noise,
            )
            losses, _, _ = rmodel.backward_step(compute_losses, hq)
            if rmodel.amp_scaler is not None:
                rmodel.amp_scaler.step(rmodel.optimizer)
                rmodel.amp_scaler.update()
            else:
                rmodel.optimizer.step()
            rmodel.model.zero_grad()
            rmodel.update_ema_model()
            step += 1

            if step % args.log_every == 0:
                hours = (time.time() - t0) / 3600
                print(f"  [epoch {epoch}] step {step}  mse={float(losses['mse'].mean()):.4f}  "
                      f"lpips={float(losses['lpips'].mean()):.4f}  {step / (hours * 3600):.2f} it/s", flush=True)
            if val is not None and step % args.eval_every_steps == 0:
                run_validation(epoch)
            if (args.max_steps and step >= args.max_steps) or (args.max_hours and time.time() - t0 >= args.max_hours * 3600):
                done = True
                break
        torch.save({"model": rmodel.model.state_dict(), "ema": rmodel.ema_state, "step": step, "epoch": epoch},
                   args.save_dir / "ressrdiff_last.pt")
        print(f"Epoch {epoch} done: step {step}, {(time.time() - t0) / 3600:.2f} h", flush=True)
        if done:
            break

    if val is not None:
        run_validation(epoch)
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"Training complete: {step} steps in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
