"""Convenience CLI to train or run inference for PF-AGCN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from train.training import main as train_main


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PF-AGCN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Launch Hydra-configured training")
    train_parser.add_argument(
        "--config-path",
        type=str,
        default="../configs",
        help="Path to Hydra config directory",
    )
    train_parser.add_argument(
        "--config-name",
        type=str,
        default="default_config",
        help="Hydra config name",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference with a trained model")
    predict_parser.add_argument("run", type=str, help="MLflow run ID or run directory")
    predict_parser.add_argument("--manifest", required=True, help="Manifest for inference")
    predict_parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Destination CSV for sigmoid scores",
    )
    predict_parser.add_argument(
        "--config",
        default="configs/default_config.yaml",
        help="Optional config to supply dataloader parameters",
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override inference batch size",
    )

    return parser.parse_args(argv)


def run_train(config_path: str, config_name: str) -> None:
    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    def _train(cfg: DictConfig) -> None:
        train_main(cfg)

    _train()


def run_predict(args: argparse.Namespace) -> None:
    from scripts.predict import main as predict_main

    predict_argv = [args.run, "--manifest", args.manifest, "--output", args.output, "--config", args.config]
    if args.batch_size is not None:
        predict_argv.extend(["--batch-size", str(args.batch_size)])
    predict_main(predict_argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "train":
        run_train(args.config_path, args.config_name)
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
