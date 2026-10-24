from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from utils import extract_state_dict


@dataclass
class AgentConfig:
    upsampler: DenoiserConfig
    num_actions: int = 0

    def __post_init__(self) -> None:
        self.upsampler.inner_model.num_actions = self.num_actions or 0


class Agent(nn.Module):
    """Container for the 2.5D EDM upsampler (keeps the ``upsampler.*`` checkpoint key prefix)."""

    def __init__(self, cfg: AgentConfig) -> None:
        super().__init__()
        self.upsampler = Denoiser(cfg.upsampler)

    @property
    def device(self):
        return self.upsampler.device

    def setup_training(self, sigma_distribution_cfg: SigmaDistributionConfig) -> None:
        self.upsampler.setup_training(sigma_distribution_cfg)

    def load(self, path_to_ckpt: Path) -> None:
        sd = torch.load(Path(path_to_ckpt), map_location=self.device, weights_only=False)
        self.upsampler.load_state_dict(extract_state_dict(sd, "upsampler"))
