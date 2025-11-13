"""Convenience CLI to train or run inference for PF-AGCN."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing import ManifestBundle, prepare_manifests
from train.training import main as train_main
from utils.system_runtime import apply_system_env

log = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PF-AGCN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Launch Hydra-configured training")
    train_parser.add_argument(
        "--config-path",
        type=str,
        default="configs",
        help="Path to Hydra config directory",
    )
    train_parser.add_argument(
        "--config-name",
        type=str,
        default="default_config",
        help="Hydra config name",
    )
    train_parser.add_argument(
        "--aspect",
        type=str,
        choices=["MF", "BP", "CC", "mf", "bp", "cc"],
        required=True,
        help="GO aspect to train (mf, bp, or cc)",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference with a trained model")
    predict_parser.add_argument("run", type=str, help="MLflow run ID or run directory")
    predict_parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest for inference (auto-generated if omitted)",
    )
    predict_parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Destination CSV for sigmoid scores",
    )
    predict_parser.add_argument(
        "--config",
        default="configs/default_config.yaml",
        help="Hydra config driving manifest discovery",
    )
    predict_parser.add_argument(
        "--aspect",
        type=str,
        choices=["MF", "BP", "CC", "mf", "bp", "cc"],
        default=None,
        help="GO aspect for inference (mf, bp, or cc)",
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override inference batch size",
    )

    return parser.parse_args(argv)


def _resolve_config_dir(config_path: str | Path) -> Path:
    """Resolve config directory relative to the project root and validate it exists."""

    config_path_str = str(config_path).strip()
    if not config_path_str:
        raise ValueError("config_path must be a non-empty string")
    normalised_path = config_path_str.replace(chr(92), "/")
    config_dir = Path(normalised_path).expanduser()
    if not config_dir.is_absolute():
        config_dir = (PROJECT_ROOT / config_dir).resolve()
    else:
        config_dir = config_dir.resolve()
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Config directory not found at {config_dir}. "
            "Set --config-path relative to the project root or provide an absolute path."
        )
    return config_dir


def _config_path_for_hydra(config_path: str | Path) -> str:
    """Hydra expects config_path relative to this script; convert and normalise separators."""

    config_dir = _resolve_config_dir(config_path)
    script_dir = Path(__file__).resolve().parent
    try:
        relative = os.path.relpath(config_dir, script_dir)
    except ValueError:
        relative = config_dir.as_posix()
    return relative.replace(chr(92), "/")


def _compose_config(
    config_dir: str | Path, config_name: str, overrides: list[str] | None = None
) -> DictConfig:
    GlobalHydra.instance().clear()
    resolved_dir = _resolve_config_dir(config_dir)
    with initialize_config_dir(
        config_dir=str(resolved_dir), job_name="pf_agcn_scripts", version_base=None
    ):
        cfg = compose(config_name=config_name, overrides=overrides or [], return_hydra_config=True)
    return cfg


def _load_manifest_meta(manifest_path: Path) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("meta", {})


def _ensure_manifests(cfg: DictConfig, aspect: str) -> List[str]:
    overrides: List[str] = []
    aspect_upper = aspect.upper()
    cfg.aspect = aspect_upper
    overrides.append(f"aspect={aspect_upper}")

    data_cfg = OmegaConf.to_container(cfg.data_config, resolve=True)
    seq_cfg = OmegaConf.to_container(cfg.model.seq_embeddings, resolve=True)
    prot_prior_cfg = OmegaConf.to_container(cfg.model.prot_prior, resolve=True)
    feature_dim = int(seq_cfg["feature_dim"])
    max_len_val = seq_cfg.get("seq_len")
    max_length = int(max_len_val) if max_len_val else None
    backend = str(seq_cfg.get("backend", "esm") or "esm").lower()

    existing_manifest = data_cfg.get("train_manifest")
    if existing_manifest:
        manifest_path = Path(existing_manifest)
        if not manifest_path.is_absolute():
            manifest_path = (PROJECT_ROOT / manifest_path).resolve()
        meta = _load_manifest_meta(manifest_path)
        manifest_aspect = str(meta.get("aspect", aspect_upper)).upper()
        if manifest_aspect != aspect_upper:
            raise ValueError(f"Manifest aspect {manifest_aspect} does not match requested {aspect_upper}")
        num_functions = meta.get("num_functions")
        if num_functions is None:
            raise ValueError(f"Manifest at {manifest_path} is missing 'num_functions' metadata")
        overrides.append(f"task.num_functions={int(num_functions)}")
        feature_dim_meta = meta.get("feature_dim")
        if feature_dim_meta is not None:
            overrides.append(f"model.seq_embeddings.feature_dim={int(feature_dim_meta)}")
        backend_meta = meta.get("embedding_backend")
        if backend_meta:
            overrides.append(f"model.seq_embeddings.backend={backend_meta}")
        if not data_cfg.get("val_manifest"):
            overrides.append(f"data_config.val_manifest={manifest_path.as_posix()}")
        if not data_cfg.get("test_manifest"):
            overrides.append(f"data_config.test_manifest={manifest_path.as_posix()}")
        return overrides

    manifest_root = (PROJECT_ROOT / "data" / "manifests").resolve()
    bundle: ManifestBundle = prepare_manifests(
        data_cfg=data_cfg,
        output_root=manifest_root,
        aspect=aspect_upper,
        feature_dim=feature_dim,
        max_length=max_length,
        protein_prior_cfg=prot_prior_cfg,
        embedding_backend=backend,
    )
    overrides.extend(
        [
            f"data_config.train_manifest={bundle.train.as_posix()}",
            f"data_config.val_manifest={bundle.val.as_posix()}",
            f"data_config.test_manifest={bundle.test.as_posix()}",
            f"task.num_functions={bundle.num_functions}",
            f"model.seq_embeddings.feature_dim={bundle.feature_dim}",
            f"model.seq_embeddings.backend={bundle.embedding_backend}",
        ]
    )
    return overrides


def run_train(config_path: str, config_name: str, aspect: str) -> None:
    aspect_upper = aspect.upper()
    hydra_config_path = _config_path_for_hydra(config_path)

    @hydra.main(version_base=None, config_path=hydra_config_path, config_name=config_name)
    def _train(cfg: DictConfig) -> None:
        apply_system_env(cfg)
        manifest_overrides = _ensure_manifests(cfg, aspect_upper)
        if manifest_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(manifest_overrides))
        train_main(cfg)

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], f"aspect={aspect_upper}"]
        _train()
    finally:
        sys.argv = original_argv


def _resolve_manifest_for_predict(args: argparse.Namespace) -> str:
    config_path = Path(args.config).resolve()
    cli_overrides: list[str] = []
    if args.aspect:
        cli_overrides.append(f"aspect={args.aspect.upper()}")
    cfg = _compose_config(config_path.parent, config_path.stem, cli_overrides)
    apply_system_env(cfg)
    data_cfg = OmegaConf.to_container(cfg.data_config, resolve=True)
    seq_cfg = OmegaConf.to_container(cfg.model.seq_embeddings, resolve=True)
    prot_prior_cfg = OmegaConf.to_container(cfg.model.prot_prior, resolve=True)
    feature_dim = int(seq_cfg["feature_dim"])
    max_len_val = seq_cfg.get("seq_len")
    max_length = int(max_len_val) if max_len_val else None
    backend = str(seq_cfg.get("backend", "esm") or "esm").lower()

    aspect = (args.aspect or cfg.get("aspect") or "").upper()

    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        meta = _load_manifest_meta(manifest_path)
        manifest_aspect = str(meta.get("aspect", aspect or "")).upper()
        if aspect and manifest_aspect and manifest_aspect != aspect:
            raise ValueError(
                f"Provided manifest aspect {manifest_aspect} does not match requested {aspect}"
            )
        return manifest_path.as_posix()

    for candidate in (
        data_cfg.get("test_manifest"),
        data_cfg.get("val_manifest"),
        data_cfg.get("train_manifest"),
    ):
        if not candidate:
            continue
        manifest_path = Path(candidate)
        if not manifest_path.is_absolute():
            manifest_path = (PROJECT_ROOT / manifest_path).resolve()
        meta = _load_manifest_meta(manifest_path)
        manifest_aspect = str(meta.get("aspect", aspect or "")).upper()
        if aspect and manifest_aspect and manifest_aspect != aspect:
            continue
        if not aspect:
            aspect = manifest_aspect
        return manifest_path.as_posix()

    if not aspect:
        raise ValueError("Aspect must be specified when auto-generating inference manifests.")

    manifest_root = (PROJECT_ROOT / "data" / "manifests").resolve()
    bundle = prepare_manifests(
        data_cfg=data_cfg,
        output_root=manifest_root,
        aspect=aspect,
        feature_dim=feature_dim,
        max_length=max_length,
        protein_prior_cfg=prot_prior_cfg,
        embedding_backend=backend,
    )
    log.info("Generated inference manifest at %s", bundle.test)
    return bundle.test.as_posix()

def run_predict(args: argparse.Namespace) -> None:
    from scripts.predict import main as predict_main

    manifest_path = _resolve_manifest_for_predict(args)
    predict_argv = [
        args.run,
        "--manifest",
        manifest_path,
        "--output",
        args.output,
        "--config",
        args.config,
    ]
    if args.batch_size is not None:
        predict_argv.extend(["--batch-size", str(args.batch_size)])
    predict_main(predict_argv)



def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv or sys.argv[1:])
    if args.command == "train":
        run_train(args.config_path, args.config_name, args.aspect)
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()

