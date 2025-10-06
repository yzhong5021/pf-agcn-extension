"""Model config loader for PF-AGCN.

Utility helpers to read YAML parameter files and build PF-AGCN instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch.nn as nn
import yaml

from model.pf_agcn import PFAGCN

ConfigPath = Union[str, Path]


def load_model_config(path: ConfigPath) -> Dict[str, Any]:
    """Parse a PF-AGCN parameter file.

    Args:
        path: Location of a YAML document describing seq_encoder, model, and
            task fields.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary with string keys.
    """
    doc = Path(path).expanduser().resolve()
    with doc.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model_from_config(
    path: ConfigPath,
    seq_encoder: nn.Module,
    *,
    seq_encoder_dim: int,
) -> PFAGCN:
    """Instantiate PF-AGCN using a parameter file.

    Args:
        path: Path to YAML with model hyperparameters.
        seq_encoder: Sequence encoder module aligned with ``seq_encoder_dim``.
        seq_encoder_dim: Width of the sequence encoder output.

    Returns:
        PFAGCN: Configured model instance ready for training or inference.
    """
    params = load_model_config(path)
    model_args = params.get("model", {})
    task_args = params.get("task", {})

    if "num_functions" not in task_args:
        raise ValueError("Config missing task.num_functions.")

    return PFAGCN(
        num_functions=int(task_args["num_functions"]),
        seq_encoder=seq_encoder,
        seq_encoder_dim=seq_encoder_dim,
        shared_dim=model_args.get("shared_dim", 128),
        graph_dim=model_args.get("graph_dim", 128),
        metric_dim=model_args.get("metric_dim", 64),
        dccn_channels=model_args.get("dccn_channels", seq_encoder_dim),
        dccn_kernel=model_args.get("dccn_kernel", 3),
        dccn_dilation=model_args.get("dccn_dilation", 2),
        dropout=model_args.get("dropout", 0.1),
        protein_steps=model_args.get("protein_steps", 2),
        function_steps=model_args.get("function_steps", 2),
        graph_keep_mass=model_args.get("graph_keep_mass", 0.9),
        graph_tau=model_args.get("graph_tau", 1.0),
    )
