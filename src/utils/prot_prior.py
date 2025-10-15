"""Protein similarity priors.

Provides helper routines for building protein-protein prior matrices used by
PF-AGCN. Two strategies are supported:

``prot_prior_blast``
    Uses the BLAST+ ``blastp`` binary via Biopython to score pairwise protein
    alignments. A binary adjacency matrix is returned where an entry is set to 1
    when the BLAST E-value for that pair falls below a prescribed threshold.

``prot_prior_esm``
    Learns a shared latent projection for ESM embeddings and produces a cosine
    distance matrix suitable for downstream graph construction.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Optional, Sequence, Tuple, Union

from Bio import SeqIO
from Bio.Application import ApplicationError
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

LOGGER = logging.getLogger(__name__)

def prot_prior_blast(
    seqs: Sequence[str],
    evalue_threshold: float = 1e-5,
    blastp_bin: str = "blastp",
) -> torch.Tensor:
    """
    Args:
        seqs: Iterable of amino-acid sequences. Each entry is treated as a BLAST query.
        evalue_threshold: E-value cutoff for retaining an alignment edge.
        blastp_bin: Name or absolute path of the blastp`` executable.

    Returns:
        (N, N) tensor with ones where the BLAST E-value is at or below the
        threshold and zeros elsewhere.

    """

    if not isinstance(seqs, Sequence):
        LOGGER.error("Error in protein prior generation: seqs must be a sequence of protein strings")

    records = []
    for idx, seq in enumerate(seqs):
        if not isinstance(seq, str):
            raise TypeError("Each sequence must be provided as a string.")
        sequence = seq.strip().upper()
        records.append(
            SeqRecord(Seq(sequence), id=f"seq_{idx}", description=f"idx={idx}")
        )

    n = len(records)

    prior = torch.eye(n, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as workdir:
        workdir_path = Path(workdir)
        query_path = workdir_path / "query.fasta"
        subject_path = workdir_path / "subject.fasta"

        for i in range(n):
            SeqIO.write([records[i]], str(query_path), "fasta")
            for j in range(i + 1, n):
                SeqIO.write([records[j]], str(subject_path), "fasta")
                cline = NcbiblastpCommandline(
                    cmd=blastp_bin,
                    query=str(query_path),
                    subject=str(subject_path),
                    outfmt="6 evalue",
                    evalue=evalue_threshold,
                    max_target_seqs=1,
                )
                try:
                    stdout, stderr = cline()
                except ApplicationError as exc:
                    raise RuntimeError(
                        "blastp execution failed"
                    ) from exc

                stderr = stderr.strip()
                if stderr:
                    raise RuntimeError(f"blastp error: {stderr}")

                found_hit = bool(stdout.strip())
                if found_hit:
                    prior[i, j] = prior[j, i] = 1.0

    return prior


class ESMProjector(nn.Module):
    """Project token embeddings into a shared latent space."""

    def __init__(self, in_dim: int, out_dim: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings) # (N,L,D) -> (N,L,out_dim)


def prot_prior_esm(
    embeddings: torch.Tensor,
    projector: Optional[nn.Module] = None,
    reduce: str = "mean",
) -> torch.Tensor:
    """Create a cosine distance prior from ESM embeddings.

    Args:
        embeddings: (N, L, D) per-residue ESM embeddings.
        projector: Provide the module for projecting embeddings to shared latent space.
        reduce: Aggregation technique; either mean or cls.

    Returns:
        (N, N) cosine distance matrix.

    """

    if embeddings.ndim != 3:
        LOGGER.error("embeddings must be a 3D tensor with shape (batch, length, dim).")
        raise ValueError(
            "embeddings must be a 3D tensor with shape (batch, length, dim)."
        )

    hidden = embeddings.size(-1)

    if projector is None:
        projector = ESMProjector(hidden)

    if reduce == "mean":
        pooled = embeddings.mean(dim=1)
    elif reduce == "cls":
        pooled = embeddings[:, 0, :]
    else:
        raise ValueError("reduce must be either 'mean' or 'cls'.")

    latent = projector(pooled)
    latent = F.normalize(latent, p=2, dim=1)
    cosine_sim = torch.matmul(latent, latent.transpose(0, 1)).clamp(-1.0, 1.0)
    cosine_dist = 1.0 - cosine_sim

    return cosine_dist


def generate_prior(
    data: Union[Sequence[str], torch.Tensor],
    method: str,
    **kwargs,
) -> torch.Tensor:
    """
    Args:
        data: Protein sequences (blast) or ESM embeddings (esm).
        method: Either blast or esm.

    Returns:
        Output of the chosen prior constructor.
    """

    if method == "blast":
        prior = prot_prior_blast(data, **kwargs)
        LOGGER.info("generated BLAST protein prior.")
        return(prior)
    elif method == "esm":
        prior = prot_prior_esm(data, **kwargs)
        LOGGER.info("generated ESM protein prior.")
        return(prior)
    else:
        raise ValueError(f"Invalid method: {method}")
