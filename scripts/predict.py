"""MLflow-backed inference helper for PF-AGCN."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import mlflow.pytorch
import numpy as np
import torch
import yaml

from modules.dataloader import build_manifest_dataloader

log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PF-AGCN inference")
    parser.add_argument("mlflow_run", type=str, help="Path to MLflow run directory or run ID")
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON/JSONL manifest with cached embeddings for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Destination CSV file for sigmoid scores",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Optional Hydra config to supply data loader parameters",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size used for inference",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    run_path = Path(args.mlflow_run)
    if run_path.is_dir():
        model_uri = run_path.as_uri()
    else:
        model_uri = f"runs:/{args.mlflow_run}/model"

    log.info("Loading PF-AGCN model from %s", model_uri)
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config_path = Path(args.config)
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    if args.batch_size is not None:
        data_cfg = dict(data_cfg)
        data_cfg["batch_size"] = args.batch_size

    dataloader = build_manifest_dataloader(
        manifest=args.manifest,
        data_cfg=data_cfg,
        base_dir=config_path.parent.resolve(),
        shuffle=False,
    )
    if dataloader is None:
        raise RuntimeError("Inference manifest is required.")

    records = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            logits = model(
                seq_embeddings=batch["seq_embeddings"],
                lengths=batch.get("lengths"),
                mask=batch.get("mask"),
                protein_prior=batch.get("protein_prior"),
                go_prior=batch.get("go_prior"),
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            records.append(probs)

    predictions = np.concatenate(records, axis=0)
    output_path = Path(args.output)
    np.savetxt(output_path, predictions, delimiter=",", fmt="%.6f")
    log.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()
