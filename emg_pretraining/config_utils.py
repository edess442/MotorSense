from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def load_config(path: str | Path) -> SimpleNamespace:
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return _to_namespace(data)


def namespace_to_dict(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [namespace_to_dict(v) for v in value]
    return value


def apply_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> SimpleNamespace:
    overrides: Dict[str, Any] = {
        "results_csv": args.results_csv,
        "run_name": args.run_name,
        "checkpoint_path": args.checkpoint_path,
        "embed_dim": args.embed_dim,
        "n_embed": args.n_embed,
        "codes_per_second": args.codes_per_second,
        "emg_window_size": args.emg_window_size,
        "experiment_dir": args.experiment_dir,
    }

    tokenizer = getattr(cfg, "tokenizer", SimpleNamespace())
    for key, value in overrides.items():
        if value is not None:
            setattr(tokenizer, key, value)
    cfg.tokenizer = tokenizer
    return cfg
