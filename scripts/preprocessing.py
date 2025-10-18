"""Utilities to build aspect-specific manifests for PF-AGCN."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.dataloader import (
    dataframe_to_multi_hot,
    parse_fasta_sequences,
    parse_ground_truth_table,
)
from utils.go_prior import Go_Prior

log = logging.getLogger(__name__)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}
ASPECT_CHOICES = {"MF", "BP", "CC"}


@dataclass
class ManifestBundle:
    """Paths and metadata for an aspect-specific manifest set."""

    aspect: str
    train: Path
    val: Path
    test: Path
    num_functions: int
    go_prior_path: Path
    terms: Sequence[str]


def _safe_filename(stem: str, registry: Dict[str, int]) -> str:
    cleaned = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in stem)
    if not cleaned:
        cleaned = "protein"
    counter = registry.setdefault(cleaned, 0)
    registry[cleaned] = counter + 1
    return f"{cleaned}_{counter}" if counter else cleaned


def _coerce_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path



def _sequence_to_embedding(sequence: str, feature_dim: int, max_length: Optional[int]) -> np.ndarray:
    tokens = sequence.strip().upper()
    if max_length is not None:
        tokens = tokens[:max_length]
    length = len(tokens)
    embedding = np.zeros((length, feature_dim), dtype=np.float32)
    for idx, aa in enumerate(tokens):
        aa_index = AA_TO_INDEX.get(aa)
        if aa_index is None or aa_index >= feature_dim:
            continue
        embedding[idx, aa_index] = 1.0
    return embedding


def _load_split_ids(split_path: Optional[str]) -> Optional[Sequence[str]]:
    if not split_path:
        return None
    path = Path(split_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "entry_id" in df.columns:
        series = df["entry_id"]
    else:
        series = df.iloc[:, 0]
    values = [str(item) for item in series.dropna().tolist()]
    return values or None


def _resolve_obo_path(go_path: Optional[Path], prior_cfg: Mapping[str, Any]) -> Path:
    candidate = go_path or _coerce_path(prior_cfg.get("obo_path"))
    if candidate and candidate.exists():
        return candidate
    raise FileNotFoundError("GO ontology (.obo) file not found in sources or go_prior config")


def _project_targets(
    entry_id: str,
    label_map: Mapping[str, Any],
    term_to_index: Mapping[str, int],
    selected_terms: Sequence[str],
) -> Sequence[float]:
    base = label_map.get(entry_id)
    if base is None:
        return [0.0] * len(selected_terms)
    values = base.tolist()
    return [float(values[term_to_index.get(term, -1)]) if term in term_to_index else 0.0 for term in selected_terms]


def prepare_manifests(
    data_cfg: Mapping[str, Any],
    *,
    output_root: Path,
    aspect: str,
    feature_dim: int,
    max_length: Optional[int] = None,
) -> ManifestBundle:
    aspect = aspect.upper()
    if aspect not in ASPECT_CHOICES:
        raise ValueError(f"Unsupported aspect '{aspect}'. Expected one of {sorted(ASPECT_CHOICES)}")

    sources = data_cfg.get("sources", {})
    if not sources:
        raise ValueError("data configuration must define 'sources'")

    raw_dir = output_root / "raw" / aspect.lower()
    raw_dir.mkdir(parents=True, exist_ok=True)

    seqs_train_path = _coerce_path(sources.get("seqs_train_path"))
    terms_train_path = _coerce_path(sources.get("terms_train_path"))
    seqs_val_path = _coerce_path(sources.get("seqs_val_path"))
    terms_val_path = _coerce_path(sources.get("terms_val_path"))
    seqs_test_path = _coerce_path(sources.get("seqs_test_path"))
    terms_test_path = _coerce_path(sources.get("terms_test_path"))

    if seqs_train_path is None or terms_train_path is None:
        raise ValueError("sources must include training sequence and term paths")
    if not seqs_train_path.exists() or not terms_train_path.exists():
        raise FileNotFoundError("Training sequence or term file missing")

    seq_tables = [parse_fasta_sequences(seqs_train_path)]
    term_tables = [parse_ground_truth_table(terms_train_path)]

    for extra_path in (seqs_val_path, seqs_test_path):
        if extra_path is not None:
            if not extra_path.exists():
                raise FileNotFoundError(f"Sequence file not found: {extra_path}")
            seq_tables.append(parse_fasta_sequences(extra_path))

    for extra_path in (terms_val_path, terms_test_path):
        if extra_path is not None:
            if not extra_path.exists():
                raise FileNotFoundError(f"Term file not found: {extra_path}")
            term_tables.append(parse_ground_truth_table(extra_path))

    sequences = (pd.concat(seq_tables, ignore_index=True)
                   .drop_duplicates(subset="entry_id", keep="first")
                   .reset_index(drop=True))

    term_table = pd.concat(term_tables, ignore_index=True)
    aggregated = term_table.groupby("entry_id")["term"].agg(list).reset_index()
    aggregated["go_terms"] = aggregated["term"].apply(json.dumps)
    split_cache = raw_dir / "train_split_cache.csv"
    aggregated[["entry_id", "go_terms"]].to_csv(split_cache, index=False)

    vocab = sorted(term_table["term"].unique())
    label_map = dataframe_to_multi_hot(term_table, vocab)
    term_to_index = {term: idx for idx, term in enumerate(vocab)}

    prior_cfg = data_cfg.get("go_prior", {})
    top_k = prior_cfg.get("top_k", {})
    split_source = prior_cfg.get("train_split_csv")
    candidate_path = _coerce_path(split_source) if split_source else None
    if candidate_path is None or not candidate_path.exists():
        if candidate_path is not None and not candidate_path.exists():
            log.warning("train_split_csv %s missing; using cached aggregate %s", candidate_path, split_cache)
        candidate_path = split_cache

    go_priors = Go_Prior(
        obo_path=_resolve_obo_path(_coerce_path(sources.get("go_path")), prior_cfg),
        train_split_csv=candidate_path,
        top_k_mf=top_k.get("MF"),
        top_k_bp=top_k.get("BP"),
        top_k_cc=top_k.get("CC"),
    )
    aspect_prior = go_priors[aspect]
    selected_terms = list(aspect_prior.terms)
    num_functions = len(selected_terms)

    priors_dir = output_root / "priors" / aspect.lower()
    priors_dir.mkdir(parents=True, exist_ok=True)
    prior_path = priors_dir / f"{aspect.lower()}_prior.npz"
    np.savez_compressed(prior_path, adjacency=aspect_prior.adjacency, terms=np.array(selected_terms))

    embeddings_dir = output_root / "embeddings" / aspect.lower()
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    registry: Dict[str, int] = {}
    record_templates: Dict[str, Dict[str, Any]] = {}
    for row in sequences.itertuples():
        entry_id = row.entry_id
        embedding = _sequence_to_embedding(row.sequence, feature_dim, max_length)
        safe_name = _safe_filename(entry_id, registry)
        embed_path = embeddings_dir / f"{safe_name}.npy"
        np.save(embed_path, embedding)
        targets = _project_targets(entry_id, label_map, term_to_index, selected_terms)
        record_templates[entry_id] = {
            "entry_id": entry_id,
            "embedding_rel": Path("embeddings") / aspect.lower() / f"{safe_name}.npy",
            "targets": targets,
            "labels": targets,
        }

    if not record_templates:
        raise RuntimeError("No records were generated while building manifests")

    split_ids = {
        "train": _load_split_ids(data_cfg.get("train_csv")),
        "val": _load_split_ids(data_cfg.get("val_csv")),
        "test": _load_split_ids(data_cfg.get("test_csv")),
    }

    meta_template = {
        "feature_dim": feature_dim,
        "max_length": max_length,
        "num_functions": num_functions,
        "terms": selected_terms,
        "aspect": aspect,
    }

    bundle_paths: Dict[str, Path] = {}
    for split, ids in split_ids.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = split_dir / f"{aspect.lower()}_manifest.json"
        selected_ids = ids or list(record_templates.keys())
        records = []
        for entry_id in selected_ids:
            template = record_templates.get(str(entry_id))
            if template is None:
                continue
            rel_embed = os.path.relpath(output_root / template["embedding_rel"], split_dir)
            rel_prior = os.path.relpath(prior_path, split_dir)
            records.append(
                {
                    "entry_id": template["entry_id"],
                    "embedding_path": rel_embed.replace(os.sep, "/"),
                    "targets": template["targets"],
                    "labels": template.get("labels", template["targets"]),
                    "go_prior_path": rel_prior.replace(os.sep, "/"),
                }
            )
        if not records:
            records = [
                {
                    "entry_id": template["entry_id"],
                    "embedding_path": os.path.relpath(output_root / template["embedding_rel"], split_dir).replace(os.sep, "/"),
                    "targets": template["targets"],
                    "labels": template.get("labels", template["targets"]),
                    "go_prior_path": os.path.relpath(prior_path, split_dir).replace(os.sep, "/"),
                }
                for template in record_templates.values()
            ]
        payload = {
            "meta": {**meta_template, "go_prior_path": os.path.relpath(prior_path, split_dir).replace(os.sep, "/")},
            "records": records,
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        bundle_paths[split] = manifest_path

    return ManifestBundle(
        aspect=aspect,
        train=bundle_paths["train"],
        val=bundle_paths["val"],
        test=bundle_paths["test"],
        num_functions=num_functions,
        go_prior_path=prior_path,
        terms=selected_terms,
    )

