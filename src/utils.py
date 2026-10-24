
from collections import OrderedDict
from functools import partial
import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from torch.optim import AdamW
import wandb


Logs = List[Dict[str, float]]
LossAndLogs = Tuple[Tensor, Dict[str, Any]]


class StateDictMixin:
    def _init_fields(self) -> None:
        def has_sd(x: str) -> bool:
            return callable(getattr(x, "state_dict", None)) and callable(getattr(x, "load_state_dict", None))

        self._all_fields = {k for k in vars(self) if not k.startswith("_")}
        self._fields_sd = {k for k in self._all_fields if has_sd(getattr(self, k))}

    def _get_field(self, k: str) -> Any:
        return getattr(self, k).state_dict() if k in self._fields_sd else getattr(self, k)

    def _set_field(self, k: str, v: Any) -> None:
        getattr(self, k).load_state_dict(v) if k in self._fields_sd else setattr(self, k, v)

    def state_dict(self) -> Dict[str, Any]:
        if not hasattr(self, "_all_fields"):
            self._init_fields()
        return {k: self._get_field(k) for k in self._all_fields}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not hasattr(self, "_all_fields"):
            self._init_fields()
        assert set(list(state_dict.keys())) == self._all_fields
        for k, v in state_dict.items():
            self._set_field(k, v)


def configure_opt(model: nn.Module, lr: float, weight_decay: float, eps: float, *blacklist_module_names: str) -> AdamW:
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTMCell, nn.LSTM)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.GroupNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif "bias" in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.endswith("weight") or pn.startswith("weight_")) and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif (pn.endswith("weight") or pn.startswith("weight_")) and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=lr, eps=eps)
    return optimizer


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def extract_state_dict(state_dict: OrderedDict, module_name: str) -> OrderedDict:
    return OrderedDict({k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def get_lr_sched(opt: torch.optim.Optimizer, num_warmup_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int):
        return 1 if current_step >= num_warmup_steps else current_step / max(1, num_warmup_steps)

    return LambdaLR(opt, lr_lambda, last_epoch=-1)


def get_path_agent_ckpt(path_ckpt_dir: Union[str, Path], epoch: int, num_zeros: int = 5) -> Path:
    d = Path(path_ckpt_dir) / "agent_versions"
    if epoch >= 0:
        return d / f"agent_epoch_{epoch:0{num_zeros}d}.pt"
    else:
        all_ = sorted(list(d.iterdir()))
        assert len(all_) >= -epoch
        return all_[epoch]


def keep_agent_copies_every(
    agent_sd: Dict[str, Any],
    epoch: int,
    path_ckpt_dir: Path,
    every: int,
    num_to_keep: Optional[int],
) -> None:
    assert every > 0
    assert num_to_keep is None or num_to_keep > 0
    get_path = partial(get_path_agent_ckpt, path_ckpt_dir)
    get_path(0).parent.mkdir(parents=False, exist_ok=True)

    # Save agent
    save_with_backup(agent_sd, get_path(epoch))

    # Clean oldest
    if (num_to_keep is not None) and (epoch % every == 0):
        get_path(max(0, epoch - num_to_keep * every)).unlink(missing_ok=True)

    # Clean previous
    if (epoch - 1) % every != 0:
        get_path(max(0, epoch - 1)).unlink(missing_ok=True)


def move_opt_to(opt: AdamW, device: torch.device):
    for optimizer_metrics in opt.state.values():
        for metric_name, metric in optimizer_metrics.items():
            if torch.is_tensor(metric) and metric_name != "step":
                optimizer_metrics[metric_name] = metric.to(device)


def save_info_for_import_script(epoch: int, run_name: str, path_ckpt_dir: Path) -> None:
    with (path_ckpt_dir / "info_for_import_script.json").open("w") as f:
        json.dump({"epoch": epoch, "name": run_name}, f)


def save_with_backup(obj: Any, path: Path):
    bk = path.with_suffix(".bk")
    if path.is_file():
        path.rename(bk)
    torch.save(obj, path)
    bk.unlink(missing_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def skip_if_run_is_over(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        path_run_is_over = Path(".run_is_over")
        if not path_run_is_over.is_file():
            func(*args, **kwargs)
            path_run_is_over.touch()
        else:
            print(f"Run is marked as finished. To unmark, remove '{str(path_run_is_over)}'.")

    return inner


def try_until_no_except(func: Callable) -> None:
    while True:
        try:
            func()
        except KeyboardInterrupt:
            break
        except Exception:
            continue
        else:
            break


def wandb_log(logs: Logs, epoch: int):
    for d in logs:
        wandb.log({"epoch": epoch, **d})
