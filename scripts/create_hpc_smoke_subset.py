"""Create a small CAFA subset for HPC smoke testing.

The script copies the first N protein sequences and associated GO term
entries from the training corpus into an HPC-friendly `smoke` directory.
Paths mirror the expectations baked into `configs/data_config/hpc_smoke.yaml`.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HPC smoke-test subset files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/orcd/home/002/lerchen/code/cafa6"),
        help="Directory containing the CAFA training data.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Directory with train_sequences.fasta and train_terms.tsv "
        "(default: <dataset-root>/Train).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for the smoke subset "
        "(default: <dataset-root>/smoke).",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=64,
        help="Number of protein entries to retain.",
    )
    parser.add_argument(
        "--sequences-name",
        type=str,
        default="train_sequences.fasta",
        help="Source FASTA filename under --train-dir.",
    )
    parser.add_argument(
        "--terms-name",
        type=str,
        default="train_terms.tsv",
        help="Source GO term TSV filename under --train-dir.",
    )
    parser.add_argument(
        "--output-sequences-name",
        type=str,
        default="train_sequences_smoke.fasta",
        help="Output FASTA filename under --output-dir.",
    )
    parser.add_argument(
        "--output-terms-name",
        type=str,
        default="train_terms_smoke.tsv",
        help="Output GO term TSV filename under --output-dir.",
    )
    return parser.parse_args()


def iter_fasta_records(path: Path) -> Iterator[Tuple[str, List[str]]]:
    header: str | None = None
    seq_lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, seq_lines
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            yield header, seq_lines


def extract_entry_id(header: str) -> str:
    cleaned = header.lstrip(">").strip()
    if not cleaned:
        raise ValueError("Encountered FASTA header without an identifier.")
    return cleaned.split()[0]


def write_fasta(records: Sequence[Tuple[str, List[str]]], dest: Path) -> None:
    with dest.open("w", encoding="utf-8") as handle:
        for header, seq_lines in records:
            handle.write(f"{header}\n")
            for line in seq_lines:
                handle.write(f"{line}\n")


def subset_terms(
    terms_path: Path,
    dest_path: Path,
    keep_ids: Iterable[str],
) -> int:
    keep = set(keep_ids)
    written = 0
    with terms_path.open("r", encoding="utf-8", newline="") as src, dest_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{terms_path} is missing a header row.")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        entry_field = reader.fieldnames[0]
        for row in reader:
            entry_id = row.get(entry_field)
            if entry_id in keep:
                writer.writerow(row)
                written += 1
    return written


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    dataset_root = args.dataset_root.expanduser().resolve()
    train_dir = (args.train_dir or dataset_root / "Train").expanduser().resolve()
    output_dir = (args.output_dir or dataset_root / "smoke").expanduser().resolve()

    sequences_src = train_dir / args.sequences_name
    terms_src = train_dir / args.terms_name
    sequences_dest = output_dir / args.output_sequences_name
    terms_dest = output_dir / args.output_terms_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Reading sequences from %s", sequences_src)
    selected_records: List[Tuple[str, List[str]]] = []
    for header, seq_lines in iter_fasta_records(sequences_src):
        selected_records.append((header, seq_lines))
        if len(selected_records) >= args.num_sequences:
            break

    if not selected_records:
        raise RuntimeError("No sequences read; check the source FASTA path.")
    if len(selected_records) < args.num_sequences:
        logging.warning(
            "Requested %d sequences but only found %d.",
            args.num_sequences,
            len(selected_records),
        )

    write_fasta(selected_records, sequences_dest)
    logging.info("Wrote %d sequences to %s", len(selected_records), sequences_dest)

    keep_ids = [extract_entry_id(header) for header, _ in selected_records]
    logging.info("Filtering terms in %s for %d EntryIDs", terms_src, len(keep_ids))
    written_terms = subset_terms(terms_src, terms_dest, keep_ids)
    logging.info("Wrote %d GO term rows to %s", written_terms, terms_dest)


if __name__ == "__main__":
    main()
