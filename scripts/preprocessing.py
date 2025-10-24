"""Utilities to build aspect-specific manifests for PF-AGCN."""

from __future__ import annotations

import hashlib
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
from utils.esm_embed import ESM_Embed
from utils.prost_embed import ProstEmbed
from utils.go_prior import Go_Prior
from utils.prot_prior import prot_prior_blast

log = logging.getLogger(__name__)

ASPECT_CHOICES = {"MF", "BP", "CC"}
EMBED_BACKENDS = {"esm", "prost"}
EMBED_CACHE_ROOTS = {
    "esm": (PROJECT_ROOT / "data" / "esm_cache").resolve(),
    "prost": (PROJECT_ROOT / "data" / "prost_cache").resolve(),
}
EMBEDDER_FACTORIES = {
    "esm": ESM_Embed,
    "prost": ProstEmbed,
}
_EMBEDDER_SINGLETONS: Dict[str, Any] = {}


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
    feature_dim: int
    embedding_backend: str


def _coerce_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _embedding_cache_path(entry_id: str, backend: str) -> Path:
    root = EMBED_CACHE_ROOTS.get(backend)
    if root is None:
        raise ValueError(f"Unsupported embedding backend '{backend}'.")
    root.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in entry_id)
    digest = hashlib.md5(entry_id.encode("utf-8")).hexdigest()[:8]
    filename = f"{safe_id or 'protein'}_{digest}.npy"
    return root / filename


def _model_cache_dir(backend: str) -> Path:
    root = EMBED_CACHE_ROOTS.get(backend)
    if root is None:
        raise ValueError(f"Unsupported embedding backend '{backend}'.")
    model_dir = (root / "models").resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _get_seq_embedder(max_length: Optional[int], backend: str):
    backend_key = backend.lower()
    if backend_key not in EMBEDDER_FACTORIES:
        raise ValueError(
            f"Unsupported embedding backend '{backend}'. Expected one of {sorted(EMBEDDER_FACTORIES)}"
        )
    embedder = _EMBEDDER_SINGLETONS.get(backend_key)
    if embedder is None:
        kwargs: Dict[str, Any] = {"cache_dir": _model_cache_dir(backend_key)}
        if backend_key == "esm" and max_length is not None:
            kwargs["max_len"] = int(max_length)
        embedder_cls = EMBEDDER_FACTORIES[backend_key]
        embedder = embedder_cls(**kwargs)
        _EMBEDDER_SINGLETONS[backend_key] = embedder
    return embedder


def _ensure_cached_embedding(
    entry_id: str,
    sequence: str,
    *,
    max_length: Optional[int],
    backend: str,
) -> tuple[Path, int, int]:
    cache_path = _embedding_cache_path(entry_id, backend)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        array = np.load(cache_path, mmap_mode="r")
        length, dim = array.shape
        return cache_path, int(length), int(dim)

    embedder = _get_seq_embedder(max_length, backend)
    embeddings, mask = embedder([sequence])
    residue_embeddings = embeddings[0][mask[0]].detach().cpu().numpy().astype(np.float32)
    np.save(cache_path, residue_embeddings)
    length, dim = residue_embeddings.shape
    return cache_path, int(length), int(dim)


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
    protein_prior_cfg: Optional[Mapping[str, Any]] = None,
    embedding_backend: str = "esm",
) -> ManifestBundle:
    aspect = aspect.upper()
    if aspect not in ASPECT_CHOICES:
        raise ValueError(f"Unsupported aspect '{aspect}'. Expected one of {sorted(ASPECT_CHOICES)}")

    backend = str(embedding_backend or "esm").lower()
    if backend not in EMBED_BACKENDS:
        raise ValueError(
            f"Unsupported embedding backend '{embedding_backend}'. Expected one of {sorted(EMBED_BACKENDS)}"
        )

    sources = data_cfg.get("sources", {})
    if not sources:
        raise ValueError("data configuration must define 'sources'")

    raw_dir = output_root / "raw" / aspect.lower()
    raw_dir.mkdir(parents=True, exist_ok=True)

    prior_cfg = dict(protein_prior_cfg or {})
    prior_enabled = bool(prior_cfg.get("enabled", False))
    prior_method = str(prior_cfg.get("method", "cosine")).lower()
    use_blast_prior = prior_enabled and prior_method == "blast"
    blast_kwargs = {
        "evalue_threshold": float(prior_cfg.get("evalue_threshold", 1e-5)),
        "blastp_bin": str(prior_cfg.get("blastp_bin", "blastp")),
    }
    blast_exec = prior_cfg.get("blastp_exec")
    if blast_exec:
        blast_kwargs["blastp_exec"] = str(blast_exec)

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

    sequences = (
        pd.concat(seq_tables, ignore_index=True)
        .drop_duplicates(subset="entry_id", keep="first")
        .reset_index(drop=True)
    )

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
            log.warning(
                "train_split_csv %s missing; using cached aggregate %s",
                candidate_path,
                split_cache,
            )
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
    np.savez_compressed(
        prior_path,
        adjacency=aspect_prior.adjacency,
        terms=np.array(selected_terms),
    )

    record_templates: Dict[str, Dict[str, Any]] = {}
    sequence_lookup: Dict[str, str] = {}
    embedding_width: Optional[int] = None
    for row in sequences.itertuples():
        entry_id = row.entry_id
        sequence_lookup[entry_id] = row.sequence
        cache_path, _, dim = _ensure_cached_embedding(
            entry_id=entry_id,
            sequence=row.sequence,
            max_length=max_length,
            backend=backend,
        )
        if embedding_width is None:
            embedding_width = dim
        elif embedding_width != dim:
            raise ValueError(
                f"Inconsistent {backend} embedding dimensionality encountered; check cache integrity."
            )
        targets = _project_targets(entry_id, label_map, term_to_index, selected_terms)
        record_templates[entry_id] = {
            "entry_id": entry_id,
            "embedding_path": cache_path,
            "targets": targets,
            "labels": targets,
        }

    if not record_templates:
        raise RuntimeError("No records were generated while building manifests")

    if embedding_width is None:
        raise RuntimeError("Failed to determine embedding dimensionality")

    if feature_dim != embedding_width:
        log.warning(
            "seq_embeddings.feature_dim=%s mismatches %s embedding dimension %s; hydra overrides will update it.",
            feature_dim,
            backend.upper(),
            embedding_width,
        )

    split_ids = {
        "train": _load_split_ids(data_cfg.get("train_csv")),
        "val": _load_split_ids(data_cfg.get("val_csv")),
        "test": _load_split_ids(data_cfg.get("test_csv")),
    }

    meta_template = {
        "feature_dim": embedding_width,
        "max_length": max_length,
        "num_functions": num_functions,
        "terms": selected_terms,
        "aspect": aspect,
        "embedding_backend": backend,
    }

    bundle_paths: Dict[str, Path] = {}
    for split, ids in split_ids.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = split_dir / f"{aspect.lower()}_manifest.json"
        selected_ids = ids or list(record_templates.keys())
        rel_go_prior = os.path.relpath(prior_path, split_dir).replace(os.sep, "/")
        records: list[Dict[str, Any]] = []
        manifest_ids: list[str] = []
        for entry_id in selected_ids:
            template = record_templates.get(str(entry_id))
            if template is None:
                continue
            manifest_ids.append(template["entry_id"])
            rel_embed = os.path.relpath(template["embedding_path"], split_dir)
            records.append(
                {
                    "entry_id": template["entry_id"],
                    "embedding_path": rel_embed.replace(os.sep, "/"),
                    "targets": template["targets"],
                    "labels": template.get("labels", template["targets"]),
                    "go_prior_path": rel_go_prior,
                }
            )
        if not records:
            records = []
            manifest_ids = []
            for template in record_templates.values():
                manifest_ids.append(template["entry_id"])
                rel_embed = os.path.relpath(template["embedding_path"], split_dir)
                records.append(
                    {
                        "entry_id": template["entry_id"],
                        "embedding_path": rel_embed.replace(os.sep, "/"),
                        "targets": template["targets"],
                        "labels": template.get("labels", template["targets"]),
                        "go_prior_path": rel_go_prior,
                    }
                )

        protein_prior_path: Optional[Path] = None
        if use_blast_prior and manifest_ids:
            sequences_for_prior: list[str] = []
            for entry_id in manifest_ids:
                seq = sequence_lookup.get(entry_id)
                if seq is None:
                    raise KeyError(
                        f"Sequence missing for entry_id '{entry_id}' required for BLAST prior"
                    )
                sequences_for_prior.append(seq)
            prior_tensor = prot_prior_blast(sequences_for_prior, **blast_kwargs)
            protein_prior_dir = (priors_dir / "protein").resolve()
            protein_prior_dir.mkdir(parents=True, exist_ok=True)
            split_prior_path = protein_prior_dir / f"{aspect.lower()}_{split}_blast_prior.npz"
            np.savez_compressed(
                split_prior_path,
                adjacency=prior_tensor.numpy(),
            )
            protein_prior_path = split_prior_path
            rel_protein_prior = os.path.relpath(split_prior_path, split_dir).replace(
                os.sep,
                "/",
            )
            for idx, record in enumerate(records):
                record["protein_prior_path"] = rel_protein_prior
                record["protein_prior_index"] = idx

        manifest_meta = {
            **meta_template,
            "go_prior_path": rel_go_prior,
        }
        if protein_prior_path is not None:
            manifest_meta["protein_prior_path"] = os.path.relpath(
                protein_prior_path, split_dir
            ).replace(os.sep, "/")
            manifest_meta["protein_prior_method"] = "blast"

        payload = {
            "meta": manifest_meta,
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
        feature_dim=embedding_width,
        embedding_backend=backend,
    )
