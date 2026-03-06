from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from .segment import SegmentId


@dataclass
class Batch:
    obs: torch.ByteTensor
    act: Union[torch.LongTensor, None]
    rew: torch.FloatTensor
    end: torch.LongTensor
    trunc: torch.LongTensor
    mask_padding: torch.BoolTensor
    info: List[Dict[str, Any]]
    segment_ids: List[SegmentId]

    def pin_memory(self) -> Batch:
        return Batch(**{k: v if k in ("segment_ids", "info") else (v.pin_memory() if v is not None else None) for k, v in self.__dict__.items()})

    def to(self, device: torch.device) -> Batch:
        return Batch(**{k: v if k in ("segment_ids", "info") else (v.to(device) if v is not None else None) for k, v in self.__dict__.items()})
