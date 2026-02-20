"""Headless inference test for diamond-depth: generates RGBD frames and saves RGB + depth."""

from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from agent import Agent
from envs import WorldModelEnv
from csgo.action_processing import encode_csgo_action, CSGOAction

OmegaConf.register_new_resolver("eval", eval)

CKPT = "outputs/2026-02-20/05-07-30/checkpoints/agent_versions/agent_epoch_00005.pt"
SPAWN_DIR = "spawn"
OUT_DIR = Path("inference_frames")
NUM_STEPS = 20


def save_rgbd(obs: torch.Tensor, path_prefix: str) -> None:
    """Save a 4-channel RGBD observation as separate RGB and depth images."""
    t = obs.add(1).div(2).mul(255).clamp(0, 255).byte().cpu()

    rgb = t[:3].permute(1, 2, 0).numpy()
    Image.fromarray(rgb).save(f"{path_prefix}_rgb.png")

    depth = t[3].numpy()
    Image.fromarray(depth).save(f"{path_prefix}_depth.png")

    side_by_side = np.zeros((rgb.shape[0], rgb.shape[1] * 2, 3), dtype=np.uint8)
    side_by_side[:, :rgb.shape[1]] = rgb
    depth_colored = np.stack([depth] * 3, axis=2)
    side_by_side[:, rgb.shape[1]:] = depth_colored
    Image.fromarray(side_by_side).save(f"{path_prefix}_sidebyside.png")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    num_actions = cfg.env.num_actions
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(Path(CKPT))
    print(f"Agent loaded. img_channels={cfg.agent.denoiser.inner_model.img_channels}")

    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)

    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(
        agent.denoiser, agent.upsampler, agent.rew_end_model,
        Path(SPAWN_DIR), 1, sl, wm_env_cfg,
        return_denoising_trajectory=False,
    )

    obs, info = wm_env.reset()
    print(f"Reset done. obs shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")

    save_rgbd(obs[0], str(OUT_DIR / "frame_000_reset"))
    print("Saved frame_000_reset (rgb + depth + sidebyside)")

    idle_action = CSGOAction([], 0, 0, False, False)
    act_tensor = encode_csgo_action(idle_action, device)

    for step in range(NUM_STEPS):
        next_obs, rew, end, trunc, info = wm_env.step(act_tensor)
        print(f"Step {step+1:3d}: obs shape {next_obs.shape}, range [{next_obs.min():.2f}, {next_obs.max():.2f}]")

        save_rgbd(next_obs[0], str(OUT_DIR / f"frame_{step+1:03d}"))

        if end or trunc:
            print("Episode ended, resetting")
            obs, info = wm_env.reset()
        else:
            obs = next_obs

    print(f"\nDone! {NUM_STEPS} frames saved to {OUT_DIR}/ (rgb, depth, sidebyside)")


if __name__ == "__main__":
    with torch.no_grad():
        main()
