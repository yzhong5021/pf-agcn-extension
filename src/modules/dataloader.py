"""
dataloader.py

Data processing utilities. Contains all dataset- and dataloader-related logic. It provides helper
functions for reading raw CAFA-format data sources (ground truths, FASTA
sequences, IA weights) as well as utilities for loading cached
tensors via manifests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

log = None
_PROTEIN_PRIOR_CACHE: Dict[Path, torch.Tensor] = {}
_ASPECT_ALIASES = {
    "C": "C",
    "CC": "C",
    "CCO": "C",
    "CELLULARCOMPONENT": "C",
    "F": "F",
    "MF": "F",
    "MFO": "F",
    "MOLECULARFUNCTION": "F",
    "M": "F",
    "P": "P",
    "BP": "P",
    "BPO": "P",
    "BIOLOGICALPROCESS": "P",
    "B": "P",
}


def _ensure_logger() -> Any:
    global log
    if log is None:
        import logging

        log = logging.getLogger(__name__)
    return log


####### RAW DATA LOADERS #######

def parse_ground_truth_table(path: Path) -> pd.DataFrame:
    """Load CAFA ground-truth annotations into a dataframe.

    Expected format: header row with columns EntryID, term, aspect.
    Columns may be separated by tabs or whitespace. Returns a dataframe
    with canonical columns: entry_id, term, aspect.
    """

    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=0,
        dtype=str,
    )
    # Map header variants to canonical names
    rename_map: Dict[str, str] = {}
    for col in list(df.columns):
        key = str(col).strip().lower()
        if key == "entryid":
            rename_map[col] = "entry_id"
        elif key == "term":
            rename_map[col] = "term"
        elif key == "aspect":
            rename_map[col] = "aspect"
    if rename_map:
        df = df.rename(columns=rename_map)
    req = {"entry_id", "term", "aspect"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"Ground-truth file {path} missing columns: {sorted(missing)}")
    # Clean values
    df["entry_id"] = df["entry_id"].astype(str).str.strip()
    df["term"] = df["term"].astype(str).str.strip()
    cleaned = df["aspect"].astype(str).str.strip().str.replace(r"[^A-Za-z]", "", regex=True).str.upper()
    mapped = cleaned.map(_ASPECT_ALIASES).fillna(cleaned)
    df["aspect"] = mapped
    df = df[df["aspect"].isin(["C", "F", "P"])].reset_index(drop=True)
    _ensure_logger().info("Loaded %d ground-truth rows from %s", len(df), path)
    return df


def parse_fasta_sequences(path: Path) -> pd.DataFrame:
    """Read a FASTA file and return protein sequences with identifiers.

    Headers are expected in the CAFA format sp|P9WHI7|RECN_MYCT; the
    *EntryID* (e.g. P9WHI7) is the second pipe-delimited token. The output
    dataframe contains three columns: entry_id (protein identifier), the
    raw header, and the amino-acid sequence.
    """

    records: Dict[str, Dict[str, str]] = {}
    current_id: Optional[str] = None
    current_header: Optional[str] = None
    sequence_chunks: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = {
                        "entry_id": current_id,
                        "header": current_header or "",
                        "sequence": "".join(sequence_chunks),
                    }
                tokens = line[1:].split("|")
                if len(tokens) < 3:
                    raise ValueError(
                        f"Unexpected FASTA header format: '{line}' (expected sp|ID|DESC)"
                    )
                current_id = tokens[1]
                current_header = line[1:]
                sequence_chunks = []
            else:
                sequence_chunks.append(line)

    if current_id is not None:
        records[current_id] = {
            "entry_id": current_id,
            "header": current_header or "",
            "sequence": "".join(sequence_chunks),
        }

    df = pd.DataFrame.from_dict(records, orient="index").reset_index(drop=True)
    _ensure_logger().info("Loaded %d sequences from %s", len(df), path)
    return df


def load_information_accretion(path: Path) -> pd.DataFrame:
    """Load IA (information accretion) scores and return a dataframe.

    The CAFA IA.txt file contains whitespace-delimited term/ia
    pairs. The dataframe can be merged with ontology tables or passed directly
    to CAFAEval. The function also normalises missing values to 0.0 for
    convenience.
    """

    df = pd.read_csv(path, sep=r"\s+", names=["term", "ia"], dtype={"term": str, "ia": float})
    df["ia"] = df["ia"].fillna(0.0)
    _ensure_logger().info("Loaded IA weights for %d terms from %s", len(df), path)
    return df


def dataframe_to_multi_hot(
    annotations: pd.DataFrame,
    vocab: Sequence[str],
    entry_id_col: str = "entry_id",
    term_col: str = "term",
) -> Dict[str, torch.Tensor]:
    """Convert an annotation dataframe to multi-hot label tensors.

    Args:
        annotations: DataFrame with at least entry_id and term columns.
        vocab: Ordered iterable of GO terms forming the target vocabulary.
        entry_id_col: Column name holding protein identifiers.
        term_col: Column name holding GO term identifiers.

    Returns:
        Mapping from entry_id to a torch.FloatTensor of shape (len(vocab),) with
        1.0 for present terms and 0.0 otherwise.
    """

    term_to_index = {term: idx for idx, term in enumerate(vocab)}
    label_map: Dict[str, torch.Tensor] = {}

    grouped = annotations.groupby(entry_id_col)[term_col].agg(list)
    for entry_id, terms in grouped.items():
        vector = torch.zeros(len(vocab), dtype=torch.float32)
        for term in terms:
            idx = term_to_index.get(term)
            if idx is not None:
                vector[idx] = 1.0
        label_map[str(entry_id)] = vector
    return label_map


class SequenceAnnotationDataset(Dataset):
    """Dataset yielding raw sequences and GO annotations.

    The dataset accepts a dataframe containing sequences (from FASTA) and a
    mapping of entry IDs to GO term lists or multi-hot vectors.
    """

    def __init__(
        self,
        sequences: pd.DataFrame,
        annotations: Mapping[str, Sequence[str]] | Dict[str, torch.Tensor],
        term_to_index: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.sequences = sequences.reset_index(drop=True)
        self.annotations = annotations
        self.term_to_index = term_to_index
        if term_to_index is not None:
            self.num_terms = len(term_to_index)
        else:
            self.num_terms = None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.sequences.iloc[index]
        entry_id = row["entry_id"]
        sample: Dict[str, Any] = {
            "entry_id": entry_id,
            "sequence": row["sequence"],
            "header": row.get("header", ""),
        }
        labels = self.annotations.get(entry_id)
        if labels is None:
            sample["terms"] = []
            if self.term_to_index is not None:
                sample["targets"] = torch.zeros(self.num_terms or 0, dtype=torch.float32)
            return sample

        if isinstance(labels, torch.Tensor):
            sample["targets"] = labels.to(dtype=torch.float32)
        else:
            sample["terms"] = list(labels)
            if self.term_to_index is not None:
                target = torch.zeros(self.num_terms or 0, dtype=torch.float32)
                for term in labels:
                    idx = self.term_to_index.get(term)
                    if idx is not None:
                        target[idx] = 1.0
                sample["targets"] = target
        return sample


###### HELPERS FOR CACHED DATA ######

def load_npz_tensor(path: Path, key: str | None = None) -> torch.Tensor:
    """Load a tensor from .npz/.npy/.pt files.

    The loader accepts numpy and torch serialisations, normalising everything to
    torch.float32 tensors for use in training.
    """

    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            return payload.to(dtype=torch.float32)
        if isinstance(payload, Mapping):
            if key is None:
                raise KeyError(
                    f"File {path} stores a mapping; please provide 'key' to select a tensor."
                )
            tensor = payload.get(key)
            if tensor is None:
                raise KeyError(f"Key '{key}' missing from {path.name}")
            return tensor.to(dtype=torch.float32)
        raise TypeError(f"Unsupported payload in {path}: {type(payload)}")

    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        return torch.from_numpy(array).to(dtype=torch.float32)

    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        try:
            default_key = "embeddings" if "embeddings" in archive.files else "arr_0"
            array_key = key or default_key
            if array_key not in archive:
                raise KeyError(
                    f"Key '{array_key}' not found in {path.name}; available={list(archive.files)}"
                )
            tensor = archive[array_key]
        finally:
            archive.close()
        return torch.from_numpy(tensor).to(dtype=torch.float32)

    raise ValueError(f"Unsupported tensor file extension: {suffix}")


def _load_cached_protein_prior(path: Path) -> torch.Tensor:
    """Load and cache a protein prior adjacency matrix."""

    resolved = path.resolve()
    cached = _PROTEIN_PRIOR_CACHE.get(resolved)
    if cached is not None:
        return cached

    archive = np.load(resolved, allow_pickle=False)
    if isinstance(archive, np.lib.npyio.NpzFile):
        if "adjacency" not in archive.files:
            raise KeyError(
                f"adjacency missing from {resolved.name}; keys={list(archive.files)}"
            )
        array = archive["adjacency"]
        archive.close()
    else:
        array = archive
    tensor = torch.as_tensor(array, dtype=torch.float32)
    _PROTEIN_PRIOR_CACHE[resolved] = tensor
    return tensor


class ManifestDataset(Dataset):
    """Dataset backed by a JSON/JSONL manifest of cached embeddings.

    Each record must provide at least a labels field (multi-hot array or
    path to a persisted tensor) and one of embedding or embedding_path.
    Optional fields include lengths, protein_prior (or
    protein_prior_path/protein_prior_index), and go_prior (or go_prior_path).
    """

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.records = self._load_manifest(manifest_path)
        if not self.records:
            raise ValueError(f"Manifest {manifest_path} did not yield any records.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        seq_embeddings = self._load_embedding(record)
        labels = self._load_tensor(record, key="labels")
        sample: Dict[str, torch.Tensor] = {
            "seq_embeddings": seq_embeddings,
            "targets": labels.to(dtype=torch.float32),
        }

        if "lengths" in record:
            sample["lengths"] = self._load_tensor(record, key="lengths").to(torch.long)

        protein_prior = self._load_optional_tensor(record, base_key="protein_prior")
        if protein_prior is not None:
            sample["protein_prior"] = protein_prior

        prior_path = record.get("protein_prior_path")
        prior_index = record.get("protein_prior_index")
        if prior_path is not None and prior_index is not None:
            resolved = Path(prior_path)
            if not resolved.is_absolute():
                resolved = (self.manifest_path.parent / resolved).resolve()
            sample["protein_prior_path"] = resolved
            sample["protein_prior_index"] = int(prior_index)

        go_prior = self._load_optional_tensor(record, base_key="go_prior")
        if go_prior is not None:
            sample["go_prior"] = go_prior

        return sample

    def _load_manifest(self, path: Path) -> list[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._read_json(path)
        raise ValueError(f"Unsupported manifest format: {path.suffix}")

    def _read_json(self, path: Path) -> list[Dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "records" in data:
            records = data["records"]
            if not isinstance(records, list):
                raise TypeError("records must be a list of manifest entries")
            return records
        raise TypeError("JSON manifest must be a list or wrap a 'records' list")

    def _load_embedding(self, record: Mapping[str, Any]) -> torch.Tensor:
        if "embedding_path" in record:
            emb_path = Path(record["embedding_path"])
            tensor = self._load_array_from_path(emb_path, key=record.get("embedding_key"))
        else:
            raise KeyError("Manifest record must include embedding info")

        if tensor.ndim != 2:
            raise ValueError("Embeddings must be 2D (length, feature_dim)")
        return tensor

    def _load_tensor(self, record: Mapping[str, Any], key: str) -> torch.Tensor:
        if key in record:
            value = record[key]
            if isinstance(value, (list, tuple)):
                return torch.tensor(value)
            if isinstance(value, (int, float)):
                return torch.tensor([value])
            if isinstance(value, str):
                return self._load_array_from_path(Path(value))
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            if isinstance(value, torch.Tensor):
                return value
            raise TypeError(f"Unsupported value type for {key}: {type(value)}")
        path_key = f"{key}_path"
        if path_key in record:
            return self._load_array_from_path(Path(record[path_key]))
        raise KeyError(f"Manifest record missing '{key}' or '{path_key}'")

    def _load_optional_tensor(
        self, record: Mapping[str, Any], base_key: str
    ) -> Optional[torch.Tensor]:
        try:
            return self._load_tensor(record, key=base_key).to(dtype=torch.float32)
        except KeyError:
            return None

    def _load_array_from_path(self, path: Path, key: Optional[str] = None) -> torch.Tensor:
        if not path.is_absolute():
            path = (self.manifest_path.parent / path).resolve()
        return load_npz_tensor(path, key)



def collate_manifest_batch(
    batch: Sequence[Dict[str, torch.Tensor]],
    protein_prior_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences and stack optional priors."""

    seqs = [item["seq_embeddings"] for item in batch]
    lengths = torch.tensor([seq.shape[0] for seq in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    targets = torch.stack([item["targets"] for item in batch])
    collated: Dict[str, torch.Tensor] = {
        "seq_embeddings": padded,
        "targets": targets,
        "lengths": lengths,
        "mask": mask,
    }

    if any("protein_prior" in item for item in batch):
        priors = [item.get("protein_prior") for item in batch]
        if all(p is not None for p in priors):
            collated["protein_prior"] = torch.stack([p for p in priors if p is not None])
    else:
        indexed_priors = [
            (item.get("protein_prior_path"), item.get("protein_prior_index"))
            for item in batch
        ]
        if all(path is not None and index is not None for path, index in indexed_priors):
            resolved_paths = []
            indices = []
            for path_value, index_value in indexed_priors:
                resolved = Path(path_value) if not isinstance(path_value, Path) else path_value
                resolved_paths.append(resolved.resolve())
                indices.append(int(index_value))
            unique_paths = set(resolved_paths)
            if len(unique_paths) != 1:
                raise ValueError(
                    "Mixed protein prior sources within a batch are not supported."
                )
            prior_matrix = _load_cached_protein_prior(unique_paths.pop())
            selector = torch.tensor(indices, dtype=torch.long)
            submatrix = prior_matrix.index_select(0, selector).index_select(1, selector)
            collated["protein_prior"] = submatrix

    if any("go_prior" in item for item in batch):
        go_priors = [item.get("go_prior") for item in batch]
        if all(p is not None for p in go_priors):
            collated["go_prior"] = go_priors[0]

    return collated


def build_manifest_dataloader(
    manifest: Optional[str],
    data_cfg: Mapping[str, Any],
    base_dir: Path,
    shuffle: bool,
    protein_prior_cfg: Optional[Mapping[str, Any]] = None,
) -> Optional[DataLoader]:
    """Create a manifest-backed dataloader if a path is provided."""

    if not manifest:
        return None
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = (base_dir / manifest_path).resolve()
    dataset = ManifestDataset(manifest_path)

    def _collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return collate_manifest_batch(batch, protein_prior_cfg=protein_prior_cfg)

    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=bool(data_cfg.get("drop_last", False)),
        collate_fn=_collate,
    )


def build_sequence_dataloader(
    sequences: pd.DataFrame,
    annotations: Mapping[str, Sequence[str]] | Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    term_to_index: Optional[Mapping[str, int]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Wrap raw sequence annotations in a DataLoader."""

    dataset = SequenceAnnotationDataset(
        sequences=sequences,
        annotations=annotations,
        term_to_index=term_to_index,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_ia_weights(cfg: Mapping[str, Any], base_dir: Path) -> Optional[np.ndarray]:
    """Resolve IA weights path from config sections and load as np.ndarray."""

    path_str = None
    if "ia_weights_path" in cfg.get("data", {}):
        path_str = cfg["data"]["ia_weights_path"]
    elif "ia_weights_path" in cfg.get("evaluation", {}):
        path_str = cfg["evaluation"]["ia_weights_path"]
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        _ensure_logger().warning("IA weights file not found at %s", path)
        return None
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        key = cfg.get("evaluation", {}).get("ia_weights_key", "weights")
        if key not in data:
            raise KeyError(
                f"IA weight key '{key}' missing from {path.name}: keys={list(data.keys())}"
            )
        weights = data[key]
    else:
        weights = data
    return np.asarray(weights, dtype=np.float32)
