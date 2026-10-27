from contextlib import contextmanager
import csv
from functools import partial
from pathlib import Path
import shutil
import time

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from agent import Agent
from metrics import psnr as compute_psnr, ssim as compute_ssim
from models.diffusion import DiffusionSampler, EDMSamplerConfig, edm_sample, upsampled_condition
from data import BatchSampler, collate_segments_to_batch, Dataset, DatasetTraverser, MRIHdf5Dataset, SegmentId
from utils import (
    configure_opt,
    count_parameters,
    get_lr_sched,
    keep_agent_copies_every,
    Logs,
    move_opt_to,
    save_info_for_import_script,
    save_with_backup,
    set_seed,
    StateDictMixin,
    try_until_no_except,
    wandb_log,
)


@contextmanager
def fixed_rng(seed: int):
    """Run a block with a fixed torch seed, then restore the global RNG state (CPU, CUDA, MPS)."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


class Trainer(StateDictMixin):
    """Trains the 2.5D slice-conditioned EDM upsampler on static MRI slice datasets."""

    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        OmegaConf.resolve(cfg)
        self._cfg = cfg
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Seed (random when common.seed is null)
        set_seed(cfg.common.seed if cfg.common.seed is not None else torch.seed() % 10 ** 9)

        # Device
        if torch.cuda.is_available():
            self._device = torch.device("cuda", self._rank)
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        print(f"Starting on {self._device}")
        self._use_cuda = self._device.type == "cuda"
        if self._use_cuda:
            torch.cuda.set_device(self._rank)  # fix compilation error on multi-gpu nodes

        self._wandb_heartbeat_interval_s = int(getattr(cfg.wandb, "heartbeat_interval_seconds", 60))

        # Init wandb
        if self._rank == 0:
            wandb_init_kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)
            assert isinstance(wandb_init_kwargs, dict)
            wandb_init_kwargs.pop("heartbeat_interval_seconds", None)
            try_until_no_except(
                partial(
                    wandb.init,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    reinit=True,
                    resume=True,
                    **wandb_init_kwargs,
                )
            )

        # Checkpointing
        self._path_ckpt_dir = Path("checkpoints")
        self._path_state_ckpt = self._path_ckpt_dir / "state.pt"
        self._keep_agent_copies = partial(
            keep_agent_copies_every,
            every=cfg.checkpointing.save_agent_every,
            path_ckpt_dir=self._path_ckpt_dir,
            num_to_keep=cfg.checkpointing.num_to_keep,
        )
        self._save_info_for_import_script = partial(
            save_info_for_import_script, run_name=cfg.wandb.name, path_ckpt_dir=self._path_ckpt_dir
        )

        # First time, init files hierarchy
        if not cfg.common.resume and self._rank == 0:
            self._path_ckpt_dir.mkdir(exist_ok=False, parents=False)
            path_config = Path("config") / "trainer.yaml"
            path_config.parent.mkdir(exist_ok=False, parents=False)
            shutil.move(".hydra/config.yaml", path_config)
            wandb.save(str(path_config))
            # Avoid copying large runtime data folders into each run output.
            def _ignore_runtime_data(_, names):
                blocked = {
                    "processed_data",
                    "processed_data_mri",
                    "raw_data",
                    "__pycache__",
                }
                return [name for name in names if name in blocked]

            shutil.copytree(src=root_dir / "src", dst="./src", ignore=_ignore_runtime_data)
            shutil.copytree(src=root_dir / "scripts", dst="./scripts")

        def _resolve_path(path_like: str | Path) -> Path:
            p = Path(path_like).expanduser()
            return p if p.is_absolute() else (root_dir / p)

        # Datasets: low-res slice episodes paired with full-res HDF5 volumes
        assert cfg.env.path_data_low_res is not None and cfg.env.path_data_full_res is not None
        dataset_full_res = MRIHdf5Dataset(_resolve_path(cfg.env.path_data_full_res))
        p = _resolve_path(cfg.env.path_data_low_res)
        num_workers = cfg.training.num_workers_data_loaders
        use_manager = cfg.training.cache_in_ram and (num_workers > 0)
        self.train_dataset = Dataset(p / "train", dataset_full_res, "train_dataset", cfg.training.cache_in_ram, use_manager)
        self.test_dataset = Dataset(p / "test", dataset_full_res, "test_dataset", cache_in_ram=True)
        self.train_dataset.load_from_default_path()
        self.test_dataset.load_from_default_path()

        # Model
        self.agent = Agent(instantiate(cfg.agent, num_actions=0)).to(self._device)
        self._upsampler = DDP(self.agent.upsampler) if dist.is_initialized() else self.agent.upsampler

        if cfg.initialization.path_to_ckpt is not None:
            self.agent.load(**cfg.initialization)

        # Optimizer and LR scheduler
        self.opt = configure_opt(self.agent.upsampler, **cfg.upsampler.optimizer)
        self.lr_sched = get_lr_sched(self.opt, cfg.upsampler.training.lr_warmup_steps)

        # Data loaders
        c = cfg.upsampler.training
        inner_cfg = cfg.agent.upsampler.inner_model
        num_next = getattr(inner_cfg, "num_next_slices_conditioning", 0)
        seq_length = inner_cfg.num_steps_conditioning + 1 + c.num_autoregressive_steps + num_next
        batch_sampler = BatchSampler(self.train_dataset, self._rank, self._world_size, c.batch_size, seq_length)
        self._data_loader_train = DataLoader(
            dataset=self.train_dataset,
            collate_fn=collate_segments_to_batch,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self._use_cuda,
            pin_memory_device=str(self._device) if self._use_cuda else "",
            batch_sampler=batch_sampler,
        )
        self._data_loader_test = DatasetTraverser(self.test_dataset, c.batch_size, seq_length)

        # Setup training
        self.agent.setup_training(instantiate(cfg.upsampler.sigma_distribution))

        # Training state (things to be saved/restored)
        self.epoch = getattr(cfg.training, "first_epoch", 1) - 1  # > 0 when continuing a finished run
        self.num_batch_train = 0
        self.num_batch_test = 0
        self.best_val_psnr = float("-inf")
        self.best_val_epoch = 0

        if cfg.common.resume:
            self.load_state_checkpoint()
        else:
            self.save_checkpoint()

        if self._rank == 0:
            print(f"{count_parameters(self.agent.upsampler)} parameters in upsampler")
            print(self.train_dataset)
            print(self.test_dataset)

    def run(self) -> None:
        num_epochs = self._cfg.training.num_final_epochs
        patience = getattr(self._cfg.training, "early_stop_patience", 0)

        if (self._rank == 0 and self._cfg.initialization.path_to_ckpt is not None and not self._cfg.common.resume
                and getattr(self._cfg.evaluation, "val_sr_every", 0)):
            # Score the starting weights, so a continued run keeps them unless training beats them.
            self.update_best_val([], self.validate_upsampler_sr()[0])

        while self.epoch < num_epochs:
            self.epoch += 1
            start_time = time.time()
            to_log = []

            if self._rank == 0:
                print(f"\nEpoch {self.epoch} / {num_epochs}\n")

            # Training
            if self._cfg.training.should:
                to_log += self.train_upsampler()

            # Evaluation
            should_test = self._rank == 0 and self._cfg.evaluation.should and (self.epoch % self._cfg.evaluation.every == 0)
            if should_test:
                test_logs = self.test_upsampler()
                to_log += test_logs
                val_sr_every = getattr(self._cfg.evaluation, "val_sr_every", 0)
                val_logs = self.validate_upsampler_sr() if val_sr_every and self.epoch % val_sr_every == 0 else []
                to_log += val_logs
                if val_logs:
                    self.update_best_val(test_logs, val_logs[0])
                to_log += self.run_inference_samples()

            # Logging
            to_log.append({"duration": (time.time() - start_time) / 3600})
            if self._rank == 0:
                wandb_log(to_log, self.epoch)

            # Checkpointing
            self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

            if patience and self.best_val_epoch and self.epoch - self.best_val_epoch >= patience:
                print(f"No new best val PSNR in {patience} epochs; stopping after epoch {self.epoch}")
                break

        if self._rank == 0:
            wandb.finish()

    def train_upsampler(self) -> Logs:
        cfg = self._cfg.upsampler.training
        if self.epoch <= cfg.start_after_epochs:
            return []
        steps = cfg.steps_first_epoch if self.epoch == 1 else cfg.steps_per_epoch

        self.agent.train()
        self.agent.zero_grad()
        model = self._upsampler
        opt = self.opt
        debug_step_logs = bool(getattr(self._cfg.training, "debug_step_logs", False))
        debug_every = int(getattr(self._cfg.training, "debug_log_every_n_steps", 1))

        torch.cuda.empty_cache()
        model.to(self._device)
        move_opt_to(opt, self._device)

        model.train()
        opt.zero_grad()
        data_iterator = iter(self._data_loader_train)
        to_log = []
        train_start = time.time()
        next_wandb_heartbeat = train_start + self._wandb_heartbeat_interval_s
        num_optimizer_updates = 0

        num_steps = cfg.grad_acc_steps * steps

        for i in trange(num_steps, desc="Training upsampler", disable=self._rank > 0):
            batch = next(data_iterator).to(self._device)
            loss, metrics = model(batch)
            loss.backward()

            metrics["num_batch_train_upsampler"] = self.num_batch_train
            self.num_batch_train += 1

            if (i + 1) % cfg.grad_acc_steps == 0:
                if cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm).item()
                    metrics["grad_norm_before_clip"] = grad_norm

                opt.step()
                opt.zero_grad()
                num_optimizer_updates += 1

                metrics["lr"] = self.lr_sched.get_last_lr()[0]
                self.lr_sched.step()

                if self._rank == 0 and self._wandb_heartbeat_interval_s > 0:
                    now = time.time()
                    if now >= next_wandb_heartbeat:
                        wandb.log(
                            {
                                "epoch": self.epoch,
                                "upsampler/train/progress": (i + 1) / max(num_steps, 1),
                                "upsampler/train/optimizer_updates": num_optimizer_updates,
                                "upsampler/train/elapsed_minutes": (now - train_start) / 60.0,
                            }
                        )
                        next_wandb_heartbeat = now + self._wandb_heartbeat_interval_s

            to_log.append(metrics)

            if debug_step_logs and self._rank == 0 and ((i + 1) % max(debug_every, 1) == 0):
                numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                print(
                    f"[debug][train][upsampler] epoch={self.epoch} step={i + 1}/{num_steps} "
                    f"loss={loss.item():.6f} obs_shape={tuple(batch.obs.shape)} metrics={numeric_metrics}"
                )

        to_log = [{f"upsampler/train/{k}": v for k, v in d.items()} for d in to_log]

        model.to("cpu")
        move_opt_to(opt, "cpu")

        return to_log

    @torch.no_grad()
    def test_upsampler(self) -> Logs:
        if self.epoch <= self._cfg.upsampler.training.start_after_epochs:
            return []
        self.agent.eval()
        model = self.agent.upsampler
        max_batches = getattr(self._cfg.evaluation, "max_test_batches", None)
        debug_step_logs = bool(getattr(self._cfg.training, "debug_step_logs", False))
        debug_every = int(getattr(self._cfg.training, "debug_log_every_n_steps", 1))
        model.eval()
        model.to(self._device)
        raw_metrics = []
        # Same sigmas and noise every epoch, so validation losses are comparable.
        with fixed_rng(getattr(self._cfg.evaluation, "val_seed", 0)):
            for i, batch in enumerate(tqdm(self._data_loader_test, desc="Evaluating upsampler")):
                if max_batches is not None and i >= max_batches:
                    break
                batch = batch.to(self._device)
                _, metrics = model(batch)
                metrics["num_batch_test_upsampler"] = self.num_batch_test
                self.num_batch_test += 1
                raw_metrics.append(metrics)

                if debug_step_logs and self._rank == 0 and ((i + 1) % max(debug_every, 1) == 0):
                    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                    print(
                        f"[debug][test][upsampler] epoch={self.epoch} batch={i + 1} "
                        f"obs_shape={tuple(batch.obs.shape)} metrics={numeric_metrics}"
                    )

        summary = {}
        if raw_metrics:
            numeric_keys = [k for k in raw_metrics[0] if isinstance(raw_metrics[0][k], (int, float))]
            for k in numeric_keys:
                vals = [m[k] for m in raw_metrics if k in m and isinstance(m[k], (int, float))]
                if vals:
                    summary[f"upsampler/test/{k}"] = sum(vals) / len(vals)

        model.to("cpu")
        return [summary] if summary else []

    @torch.no_grad()
    def validate_upsampler_sr(self) -> Logs:
        """Mean PSNR/SSIM of evenly spaced validation slices, sampled with the shared EDM sampler and a fixed seed."""
        cfg = self._cfg.evaluation
        upsampler = self.agent.upsampler
        upsampler.eval().to(self._device)
        n_cond = upsampler.cfg.inner_model.num_steps_conditioning
        n_next = upsampler.cfg.inner_model.num_next_slices_conditioning
        up_factor = upsampler.cfg.upsampling_factor
        up_mode = "bilinear" if self._device.type == "mps" else "bicubic"
        sampler_cfg = EDMSamplerConfig(num_steps=getattr(cfg, "val_sr_num_steps", 20))
        generator = torch.Generator(device=self._device).manual_seed(getattr(cfg, "val_seed", 0))
        slices_per_volume = getattr(cfg, "val_sr_slices_per_volume", 16)

        psnrs, ssims = [], []
        for ep_id in tqdm(range(self.test_dataset.num_episodes), desc="Validating upsampler (sampling)"):
            episode = self.test_dataset.load_episode(ep_id)
            obs_lr = episode.obs.float().div(255).mul(2).sub(1) if episode.obs.dtype == torch.uint8 else episode.obs.float()
            length = obs_lr.size(0)
            slice_ids = torch.linspace(0, length - 1, min(slices_per_volume, length)).round().long().unique().tolist()
            for si in slice_ids:
                cond_frames = upsampled_condition(obs_lr.to(self._device), si, n_cond, n_next, up_factor, up_mode)
                _, t, c, h_up, w_up = cond_frames.shape
                cond = cond_frames.reshape(1, t * c, h_up, w_up)
                sr = edm_sample(
                    lambda x, sigma: upsampler.denoise(x, sigma, None, cond, None),
                    torch.Size((1, c, h_up, w_up)), sampler_cfg, self._device, generator,
                )
                hr = self.test_dataset[SegmentId(ep_id, si, si + 1)].info["full_res"][:1].to(self._device)
                psnrs.append(compute_psnr(sr, hr))
                ssims.append(compute_ssim(sr, hr))

        upsampler.to("cpu")
        if not psnrs:
            return []
        return [{
            "upsampler/val/psnr": float(np.mean(psnrs)),
            "upsampler/val/ssim": float(np.mean(ssims)),
            "upsampler/val/num_slices": len(psnrs),
        }]

    def update_best_val(self, test_logs: Logs, val_log: dict) -> None:
        """Save the agent as agent_best_val.pt when validation PSNR improves; append to val_metrics.csv."""
        val_psnr = val_log["upsampler/val/psnr"]
        is_best = val_psnr > self.best_val_psnr
        if is_best:
            self.best_val_psnr = val_psnr
            self.best_val_epoch = self.epoch
            torch.save(self.agent.state_dict(), self._path_ckpt_dir / "agent_best_val.pt")
            print(f"New best val PSNR {val_psnr:.2f} dB at epoch {self.epoch}; saved agent_best_val.pt")

        path_csv = self._path_ckpt_dir / "val_metrics.csv"
        test_summary = test_logs[0] if test_logs else {}
        row = {
            "epoch": self.epoch,
            "num_batch_train": self.num_batch_train,
            "val_loss_denoising": test_summary.get("upsampler/test/loss_denoising", float("nan")),
            "val_psnr": val_psnr,
            "val_ssim": val_log["upsampler/val/ssim"],
            "is_best": int(is_best),
        }
        write_header = not path_csv.is_file()
        with path_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @torch.no_grad()
    def run_inference_samples(self) -> Logs:
        """Super-resolve a few test slices and log GT / generated / bicubic grids to W&B."""
        cfg = self._cfg
        num_samples = cfg.evaluation.num_inference_samples
        up_mode = "bilinear" if self._device.type == "mps" else "bicubic"

        upsampler = self.agent.upsampler
        upsampler.eval().to(self._device)
        sampler_up = DiffusionSampler(upsampler, instantiate(cfg.sampler))
        n_cond_up = upsampler.cfg.inner_model.num_steps_conditioning
        n_next_up = upsampler.cfg.inner_model.num_next_slices_conditioning
        up_factor = upsampler.cfg.upsampling_factor

        seq_length = n_cond_up + 1 + cfg.evaluation.num_inference_steps
        traverser = DatasetTraverser(self.test_dataset, 1, seq_length)

        to_log = []
        sample_count = 0

        def to_grayscale_or_rgb(t):
            img = t.clamp(-1, 1).add(1).div(2).mul(255).byte().cpu()
            if img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            elif img.shape[0] >= 4:
                img = img[:3]
            return img.permute(1, 2, 0).numpy()

        def make_grid(row_pairs):
            h, w, _ = row_pairs[0][0].shape
            gap = 2
            n_cols = len(row_pairs[0])
            n_rows = len(row_pairs)
            grid_w = n_cols * w + (n_cols - 1) * gap
            grid_h = n_rows * h + (n_rows - 1) * gap
            grid = np.full((grid_h, grid_w, 3), 128, dtype=np.uint8)
            for r, row in enumerate(row_pairs):
                y = r * (h + gap)
                for c_idx, img in enumerate(row):
                    x = c_idx * (w + gap)
                    grid[y:y + h, x:x + w] = img
            return grid

        for batch in traverser:
            if sample_count >= num_samples:
                break

            batch = batch.to(self._device)
            has_hr = bool(batch.info and isinstance(batch.info[0], dict) and "full_res" in batch.info[0])
            if not has_hr:
                continue

            obs_lr = batch.obs
            obs_hr = batch.info[0]["full_res"].to(self._device)
            if obs_hr.ndim == 4:
                obs_hr = obs_hr.unsqueeze(0)

            b, t, c, h, w = obs_lr.shape
            vis_count = min(6, t)
            vis_idx = [int(i) for i in torch.linspace(0, t - 1, vis_count).long()]

            gen_hr_frames = []
            for i in vis_idx:
                prev = upsampled_condition(obs_lr[0], i, n_cond_up, n_next_up, up_factor, up_mode)
                sr_frame, _ = sampler_up.sample(prev, None)
                gen_hr_frames.append(sr_frame[0])

            gt_imgs = [to_grayscale_or_rgb(obs_hr[0, i]) for i in vis_idx]
            gen_imgs = [to_grayscale_or_rgb(f) for f in gen_hr_frames]
            lr_imgs = [to_grayscale_or_rgb(
                F.interpolate(obs_lr[:, i], scale_factor=up_factor, mode=up_mode)[0]
            ) for i in vis_idx]

            grid = make_grid([gt_imgs, gen_imgs, lr_imgs])
            frame_labels = ", ".join(str(i) for i in vis_idx)
            to_log.append({
                f"inference/sample_{sample_count}": wandb.Image(
                    grid, caption=f"Top: GT HR | Mid: Generated HR | Bottom: Bicubic LR | slices: {frame_labels}",
                ),
            })
            sample_count += 1

        upsampler.to("cpu")
        return to_log

    def load_state_checkpoint(self) -> None:
        self.load_state_dict(torch.load(self._path_state_ckpt, map_location=self._device, weights_only=False))

    def save_checkpoint(self) -> None:
        if self._rank == 0:
            save_with_backup(self.state_dict(), self._path_state_ckpt)
            self.train_dataset.save_to_default_path()
            self.test_dataset.save_to_default_path()
            self._keep_agent_copies(self.agent.state_dict(), self.epoch)
            self._save_info_for_import_script(self.epoch)
