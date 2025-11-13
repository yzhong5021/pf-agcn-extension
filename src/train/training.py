"""Training entry point for PF-AGCN using PyTorch Lightning.

Hydra-configured trainer with MLflow tracking and CAFA metric logging.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import hydra
from hydra.utils import get_original_cwd
import mlflow
import mlflow.pytorch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from cafaeval.evaluation import compute_metrics as cafa_compute_metrics
from cafaeval.evaluation import normalize as cafa_normalize
from cafaeval.graph import GroundTruth as CafaGroundTruth
from cafaeval.graph import Prediction as CafaPrediction

from src.model.model import PFAGCN
from src.modules.dataloader import build_manifest_dataloader, load_ia_weights
from src.modules.loss import BCEWithLogits
from src.utils.system_runtime import apply_system_env, merged_mlflow_settings

log = logging.getLogger(__name__)


###### Config utils ######


def flatten_config(config: Mapping[str, Any], parent: str = "") -> Dict[str, Any]:
    """Flatten nested mappings to MLflow-friendly key/value pairs."""

    items: Dict[str, Any] = {}
    for key, value in config.items():
        composite = f"{parent}.{key}" if parent else str(key)
        if isinstance(value, Mapping):
            items.update(flatten_config(value, composite))
        elif isinstance(value, (list, tuple)):
            items[composite] = json.dumps(value)
        else:
            items[composite] = value
    return items


def to_namespace(data: Any) -> Any:
    """Recursively convert dictionaries to SimpleNamespace objects."""

    if isinstance(data, Mapping):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in data.items()})
    if isinstance(data, list):
        return [to_namespace(v) for v in data]
    return data


def build_model_config(cfg: DictConfig) -> Any:
    """Create a model config object compatible with PFAGCN."""

    container = {
        "task": OmegaConf.to_container(cfg.task, resolve=True),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
    }
    try:
        from model import config as model_config

        task_cls = getattr(model_config, "TaskConfig")
        model_cls = getattr(model_config, "PFAGCNModelConfig")
        config_cls = getattr(model_config, "PFAGCNConfig")

        task_cfg = task_cls(**container["task"])
        model_cfg = model_cls(**container["model"])
        return config_cls(task=task_cfg, model=model_cfg)
    except (ImportError, AttributeError, TypeError):
        log.debug("Falling back to SimpleNamespace-based model config")
        return to_namespace(container)


def build_model(cfg: DictConfig) -> PFAGCN:
    """Instantiate the PF-AGCN model."""

    model_config = build_model_config(cfg)
    return PFAGCN(model_config)


def build_loss(cfg: DictConfig) -> nn.Module:
    """Instantiate the configured criterion."""

    loss_cfg = cfg.model.loss
    name = str(loss_cfg.name).lower()
    if name == "bce_with_logits":
        pos_weight = loss_cfg.pos_weight
        tensor_weight = None
        if pos_weight is not None:
            tensor_weight = torch.tensor(pos_weight, dtype=torch.float32)
        return BCEWithLogits(pos_weight=tensor_weight)
    raise ValueError(f"Unsupported loss: {loss_cfg.name}")


def build_optimizer(cfg: DictConfig, parameters: Iterable[nn.Parameter]) -> Optimizer:
    """Create the optimiser defined in the config."""

    optim_cfg = cfg.optimizer
    name = str(optim_cfg.name).lower()
    if name == "adamw":
        betas = tuple(optim_cfg.betas) if optim_cfg.get("betas") else (0.9, 0.999)
        return torch.optim.AdamW(
            parameters,
            lr=float(optim_cfg.lr),
            betas=betas,
            weight_decay=float(optim_cfg.weight_decay),
        )
    if name == "adam":
        betas = tuple(optim_cfg.betas) if optim_cfg.get("betas") else (0.9, 0.999)
        return torch.optim.Adam(
            parameters,
            lr=float(optim_cfg.lr),
            betas=betas,
            weight_decay=float(optim_cfg.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer: {optim_cfg.name}")


def build_scheduler(cfg: DictConfig, optimizer: Optimizer) -> Optional[LambdaLR]:
    """Configure learning-rate scheduling with optional warmup."""

    sched_cfg = cfg.scheduler
    name = str(sched_cfg.name).lower()
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {sched_cfg.name}")

    total_epochs = int(cfg.training.max_epochs)
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
    min_lr = float(sched_cfg.get("min_lr", 0.0))
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def make_lambda(base_lr: float):
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(current_epoch: int) -> float:
            if warmup_epochs > 0 and current_epoch < warmup_epochs:
                return (current_epoch + 1) / float(max(1, warmup_epochs))
            progress = (current_epoch - warmup_epochs) / float(
                max(1, total_epochs - warmup_epochs)
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return lr_lambda

    lambdas = [make_lambda(lr) for lr in base_lrs]
    return LambdaLR(optimizer, lr_lambda=lambdas)


######### CAFA metrics ###########


def compute_cafa_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: Sequence[float],
    ia_weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute CAFA metrics via cafaeval utilities."""

    if probabilities.size == 0 or targets.size == 0:
        return {
            "fmax": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "pr_auc": 0.0,
            "ap": 0.0,
            "ia_fmax": 0.0,
            "ia_precision": 0.0,
            "ia_recall": 0.0,
            "ia_threshold": thresholds[0] if thresholds else 0.5,
        }

    tau_arr = np.asarray(list(thresholds) if thresholds else [0.5], dtype=np.float32)
    ids = {str(idx): idx for idx in range(probabilities.shape[0])}
    prediction = CafaPrediction(ids=ids, matrix=probabilities.astype(np.float32))
    ground_truth = CafaGroundTruth(ids=ids, matrix=targets.astype(bool))
    toi = np.arange(probabilities.shape[1])
    ne = np.full(tau_arr.shape[0], ground_truth.matrix.shape[0])

    metrics_df = cafa_normalize(
        cafa_compute_metrics(
            prediction,
            ground_truth,
            tau_arr,
            toi,
            ic_arr=None,
            n_cpu=1,
        ),
        "mock",
        tau_arr,
        ne,
        normalization="cafa",
    )
    metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)
    metrics_df = metrics_df.dropna(subset=["f"], how="all")

    if metrics_df.empty:
        best_precision = best_recall = best_fmax = 0.0
        best_tau = float(tau_arr[0])
    else:
        best_idx = metrics_df["f"].astype(float).idxmax()
        best_row = metrics_df.loc[best_idx]
        best_precision = float(best_row.get("pr", 0.0))
        best_recall = float(best_row.get("rc", 0.0))
        best_fmax = float(best_row.get("f", 0.0))
        best_tau = float(best_row.get("tau", tau_arr[0]))

    ia_fmax = best_fmax
    ia_precision = best_precision
    ia_recall = best_recall
    ia_threshold = best_tau

    if ia_weights is not None:
        ia_df = cafa_normalize(
            cafa_compute_metrics(
                prediction,
                ground_truth,
                tau_arr,
                toi,
                ic_arr=ia_weights,
                n_cpu=1,
            ),
            "mock",
            tau_arr,
            ne,
            normalization="cafa",
        )
        ia_df = ia_df.replace([np.inf, -np.inf], np.nan)
        ia_df = ia_df.dropna(subset=["f"], how="all")
        if not ia_df.empty:
            ia_best_idx = ia_df["f"].astype(float).idxmax()
            ia_row = ia_df.loc[ia_best_idx]
            ia_fmax = float(ia_row.get("f", ia_fmax))
            ia_precision = float(ia_row.get("pr", ia_precision))
            ia_recall = float(ia_row.get("rc", ia_recall))
            ia_threshold = float(ia_row.get("tau", ia_threshold))

    y_true = targets.reshape(-1)
    y_scores = probabilities.reshape(-1)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_curve, precision_curve)
    ap_micro = average_precision_score(y_true, y_scores)

    roc_auc: Optional[float]
    try:
        roc_auc = float(roc_auc_score(targets, probabilities, average="macro"))
    except ValueError:
        roc_auc = None

    metrics = {
        "fmax": best_fmax,
        "precision": best_precision,
        "recall": best_recall,
        "pr_auc": pr_auc,
        "ap": ap_micro,
        "ia_fmax": ia_fmax,
        "ia_precision": ia_precision,
        "ia_recall": ia_recall,
        "ia_threshold": ia_threshold,
    }
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
    return metrics


####### Lightning Module #######

class PFAGCNLightningModule(LightningModule):
    """Lightning wrapper around PF-AGCN with CAFA evaluation."""

    def __init__(
        self,
        cfg: DictConfig,
        thresholds: Sequence[float],
        ia_weights: Optional[np.ndarray],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.thresholds = list(thresholds)
        self.ia_weights = ia_weights
        self.model = build_model(cfg)
        self.criterion = build_loss(cfg)
        self.best_ia_fmax = -float("inf")
        self._val_probs: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._val_losses: List[torch.Tensor] = []

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(
            seq_embeddings=batch["seq_embeddings"],
            lengths=batch.get("lengths"),
            mask=batch.get("mask"),
            protein_prior=batch.get("protein_prior"),
            go_prior=batch.get("go_prior"),
        )
        return outputs.logits

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:  # noqa: ARG002
        logits = self.forward(batch)
        loss = self.criterion(logits, batch["targets"])
        batch_size = batch["targets"].size(0)
        self.log(
            "train/loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=False,
        )
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=self._sync_dist,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_probs = []
        self._val_targets = []
        self._val_losses = []

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:  # noqa: ARG002
        logits = self.forward(batch)
        loss = self.criterion(logits, batch["targets"])
        probabilities = torch.sigmoid(logits)
        self._val_probs.append(probabilities.detach().cpu())
        self._val_targets.append(batch["targets"].detach().cpu())
        self._val_losses.append(loss.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_probs:
            return
        probs_np = torch.cat(self._val_probs).numpy()
        targets_np = torch.cat(self._val_targets).numpy()
        metrics = compute_cafa_metrics(
            probabilities=probs_np,
            targets=targets_np,
            thresholds=self.thresholds,
            ia_weights=self.ia_weights,
        )
        mean_loss = float(torch.stack(self._val_losses).mean().item())

        self.log(
            "val/loss",
            mean_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        self.log(
            "cafa/ia_fmax",
            metrics["ia_fmax"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        self.log(
            "cafa/threshold",
            metrics["ia_threshold"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "cafa/precision",
            metrics["ia_precision"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "cafa/recall",
            metrics["ia_recall"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "cafa/pr_auc",
            metrics["pr_auc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "cafa/ap",
            metrics["ap"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        if "roc_auc" in metrics:
            self.log(
                "cafa/roc_auc",
                metrics["roc_auc"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        self.best_ia_fmax = max(self.best_ia_fmax, metrics["ia_fmax"])

    def configure_optimizers(self) -> Any:
        optimizer = build_optimizer(self.cfg, self.parameters())
        scheduler = build_scheduler(self.cfg, optimizer)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @property
    def _sync_dist(self) -> bool:
        return bool(self.trainer and self.trainer.num_devices > 1)


# ---------------------------------------------------------------------------
# MLflow integrations
# ---------------------------------------------------------------------------


class MLflowModelSaver(Callback):
    """Callback to log the trained model to MLflow once per run."""

    def __init__(self) -> None:
        super().__init__()
        self.logged = False

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.logged:
            return
        logger = trainer.logger
        if not isinstance(logger, MLFlowLogger):
            return
        run_id = logger.run_id
        tracking_uri = logger._tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        active_run = mlflow.active_run()
        started_run = False
        if active_run is None or active_run.info.run_id != run_id:
            mlflow.start_run(run_id=run_id)
            started_run = True
        try:
            mlflow.pytorch.log_model(pl_module.model, name="model")
        finally:
            if started_run:
                mlflow.end_run()
        self.logged = True


def _prepare_mlflow_logger(cfg: DictConfig, base_dir: Path) -> MLFlowLogger:
    mlflow_cfg = merged_mlflow_settings(cfg)
    tracking_uri = mlflow_cfg.get("tracking_uri")
    artifact_location = mlflow_cfg.get("artifact_root")
    if not tracking_uri:
        tracking_dir = (base_dir / "mlruns").resolve()
        tracking_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_dir.as_posix()}"
        artifact_location = artifact_location or tracking_uri
    logger = MLFlowLogger(
        experiment_name=mlflow_cfg.get("experiment_name", cfg.get("experiment_name", "pfagcn")),
        tracking_uri=tracking_uri,
        run_name=mlflow_cfg.get("run_name", cfg.get("run_name", "pf-agcn")),
        artifact_location=artifact_location,
    )
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger.log_hyperparams(flatten_config(resolved_cfg))
    logger.experiment.log_dict(logger.run_id, resolved_cfg, artifact_file="hydra_config.json")
    return logger


def _precision_arg(cfg: DictConfig) -> Any:
    precision_cfg = int(cfg.training.get("precision", 32))
    if precision_cfg == 16:
        return "16-mixed"
    return precision_cfg


###### Hydra main ######


@hydra.main(version_base=None, config_path="../../configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    """Hydra application entry point."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    apply_system_env(cfg)
    base_dir = Path(get_original_cwd())
    seed_everything(int(cfg.training.get("seed", 42)), workers=True)
    prot_prior_cfg = OmegaConf.to_container(
        getattr(cfg.model, "prot_prior", {}), resolve=True
    )

    train_loader = build_manifest_dataloader(
        cfg.data_config.get("train_manifest"),
        cfg.data_config,
        base_dir,
        shuffle=True,
        protein_prior_cfg=prot_prior_cfg,
    )
    if train_loader is None:
        raise RuntimeError("Training manifest is required to start training.")
    val_loader = build_manifest_dataloader(
        cfg.data_config.get("val_manifest"),
        cfg.data_config,
        base_dir,
        shuffle=False,
        protein_prior_cfg=prot_prior_cfg,
    )

    thresholds = list(cfg.evaluation.get("threshold_grid", [0.5]))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ia_weights = load_ia_weights(cfg_dict, base_dir)

    model = PFAGCNLightningModule(cfg=cfg, thresholds=thresholds, ia_weights=ia_weights)
    mlflow_logger = _prepare_mlflow_logger(cfg, base_dir)

    callbacks = [
        ModelCheckpoint(
            monitor="cafa/ia_fmax",
            mode="max",
            filename="epoch{epoch:03d}-iafmax{cafa/ia_fmax:.4f}",
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        MLflowModelSaver(),
    ]

    trainer_kwargs: Dict[str, Any] = {
        "logger": mlflow_logger,
        "max_epochs": int(cfg.training.max_epochs),
        "precision": _precision_arg(cfg),
        "gradient_clip_val": float(cfg.training.get("gradient_clip", 0.0)),
        "accumulate_grad_batches": int(cfg.training.get("accumulate_batches", 1)),
        "log_every_n_steps": int(cfg.training.get("log_interval", 50)),
        "check_val_every_n_epoch": int(cfg.evaluation.get("val_interval", 1)),
        "callbacks": callbacks,
        "default_root_dir": str(base_dir),
    }

    optional_keys = {
        "devices": "devices",
        "accelerator": "accelerator",
        "strategy": "strategy",
        "num_nodes": "num_nodes",
        "deterministic": "deterministic",
        "enable_progress_bar": "enable_progress_bar",
        "limit_train_batches": "limit_train_batches",
        "limit_val_batches": "limit_val_batches",
        "fast_dev_run": "fast_dev_run",
    }
    for cfg_key, trainer_key in optional_keys.items():
        if cfg.training.get(cfg_key) is not None:
            trainer_kwargs[trainer_key] = cfg.training[cfg_key]

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()


