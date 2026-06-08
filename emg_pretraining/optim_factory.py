# optim/factory.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _default_layer_id_from_name(name: str) -> int:
    """
    Best-effort mapping for hierarchical ViT-like models:
    - embeddings/stem -> 0
    - blocks.N.* -> N+1
    - head -> last
    """
    lname = name.lower()
    if any(k in lname for k in ["patch_embed", "stem", "pos_embed", "cls_token"]):
        return 0
    if "blocks." in lname:
        try:
            parts = lname.split("blocks.")[1]
            idx = int(parts.split(".")[0])
            return idx + 1
        except Exception:
            return 1
    return 9999


def make_param_groups_layerwise(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
    no_decay_keys: Tuple[str, ...] = (
        "bias",
        "norm",
        "ln",
        "bn",
        "pos_embed",
        "cls_token",
    ),
) -> List[Dict[str, Any]]:
    """
    Creates param groups with per-layer lr scaling and wd separation.
    """
    params = []
    max_layer = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = _default_layer_id_from_name(name)
        max_layer = max(max_layer, lid)
        params.append((name, p, lid))

    # layer -> lr scale (earlier layers smaller lr)
    layer_scales = {}
    for lid in range(max_layer + 1):
        depth_from_end = max_layer - lid
        layer_scales[lid] = layer_decay ** depth_from_end

    groups: Dict[Tuple[int, bool], Dict[str, Any]] = {}

    for name, p, lid in params:
        is_no_decay = any(k in name.lower() for k in no_decay_keys)
        key = (lid, is_no_decay)

        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * layer_scales.get(lid, 1.0),
                "weight_decay": 0.0 if is_no_decay else weight_decay,
            }

        groups[key]["params"].append(p)

    return list(groups.values())

def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    ocfg = cfg.optim.optimizer
    name = ocfg.name.lower()

    weight_decay = float(getattr(ocfg, "weight_decay", 0.0))
    lr_backbone = float(getattr(ocfg, "lr_backbone", getattr(ocfg, "lr", 0.00002)))
    lr_head = float(getattr(ocfg, "lr_head", getattr(ocfg, "lr", 0.0002)))

    param_groups = []
    for name_param, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in name_param.lower():
            lr_use = lr_backbone
        else:
            lr_use = lr_head
        param_groups.append({
            'params': [p],
            'lr': lr_use,
            'weight_decay': weight_decay,
        })

    assert len(param_groups) > 0, "Model has no trainable parameters"

    if name == "adamw":
        betas = tuple(getattr(ocfg, "betas", (0.9, 0.999)))
        eps = float(getattr(ocfg, "eps", 1e-8))
        return torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    if name == "sgd":
        momentum = float(getattr(ocfg, "momentum", 0.9))
        nesterov = bool(getattr(ocfg, "nesterov", False))
        return torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

    raise ValueError(f"Unknown optimizer: {ocfg.name}")


# def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
#     ocfg = cfg.optim.optimizer
#     name = ocfg.name.lower()

#     base_lr = float(ocfg.lr)
#     weight_decay = float(getattr(ocfg, "weight_decay", 0.0))

#     # optional layer-wise lr decay
#     layer_wise_decay = getattr(ocfg, "layer_wise_decay", None)
#     if layer_wise_decay is not None and float(layer_wise_decay) > 0:
#         param_groups = make_param_groups_layerwise(
#             model=model,
#             base_lr=base_lr,
#             weight_decay=weight_decay,
#             layer_decay=float(layer_wise_decay),
#         )
#     else:
#         param_groups = [{
#             "params": [p for p in model.parameters() if p.requires_grad],
#             "lr": base_lr,
#             "weight_decay": weight_decay,
#         }]

#     if name == "adamw":
#         betas = tuple(getattr(ocfg, "betas", (0.9, 0.999)))
#         eps = float(getattr(ocfg, "eps", 1e-8))
#         return torch.optim.AdamW(
#             param_groups,
#             lr=base_lr,
#             betas=betas,
#             eps=eps,
#         )

#     if name == "sgd":
#         momentum = float(getattr(ocfg, "momentum", 0.9))
#         nesterov = bool(getattr(ocfg, "nesterov", False))
#         return torch.optim.SGD(
#             param_groups,
#             lr=base_lr,
#             momentum=momentum,
#             weight_decay=weight_decay,
#             nesterov=nesterov,
#         )

#     raise ValueError(f"Unknown optimizer: {ocfg.name}")


# def build_scheduler(cfg, optimizer):
#     if not hasattr(cfg.optim, "scheduler") or cfg.optim.scheduler is None:
#         return None

#     sched_cfg = cfg.optim.scheduler

#     if sched_cfg.name == "multistep":
#         return torch.optim.lr_scheduler.MultiStepLR(
#             optimizer,
#             milestones=sched_cfg.milestones,
#             gamma=sched_cfg.gamma,
#         )

#     raise ValueError(f"Unknown scheduler: {sched_cfg.name}")