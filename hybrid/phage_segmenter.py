#!/usr/bin/env python3
"""Segment phage genomes using a DS-HDP-HMM with hybrid emissions."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hybrid.emissions import (
    BetaBinomialPrior,
    GaussianNIWPrior,
    HybridEmissionModel,
)
from hybrid.sampler import DSHDPHMMHybridSampler, HybridObservations


@dataclass
class PhageRecord:
    """Container with derived attributes for a single ORF."""

    gene: str
    contig: str
    start: int
    end: int
    length: int
    strand: int
    gc_fraction: float
    gc_successes: int
    category: str
    category_encoded: int
    rbs: str
    rbs_encoded: int
    intergenic_up: float
    intergenic_down: float
    overlap_up: int
    overlap_down: int


class LabelEncoder:
    """Simple label encoder that preserves insertion order."""

    def __init__(self) -> None:
        self._to_index: Dict[str, int] = {}
        self._labels: List[str] = []

    def encode(self, label: str) -> int:
        if label not in self._to_index:
            self._to_index[label] = len(self._labels)
            self._labels.append(label)
        return self._to_index[label]

    @property
    def labels(self) -> List[str]:
        return list(self._labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment phage genomes with a DS-HDP-HMM using hybrid emissions.",
    )
    parser.add_argument("input", type=Path, help="Input TSV with ORF annotations")
    parser.add_argument("output", type=Path, help="Output TSV with segment summaries")
    parser.add_argument("--iterations", type=int, default=300, help="Number of Gibbs samples")
    parser.add_argument("--burn-in", type=int, default=150, help="Burn-in iterations")
    parser.add_argument("--alpha", type=float, default=6.0, help="Sticky HDP alpha parameter")
    parser.add_argument("--gamma", type=float, default=4.0, help="Global concentration gamma")
    parser.add_argument("--rho0", type=float, default=5.0, help="Beta prior parameter rho0")
    parser.add_argument("--rho1", type=float, default=1.0, help="Beta prior parameter rho1")
    parser.add_argument("--beta-alpha", type=float, default=1.5, help="Beta prior alpha for GC")
    parser.add_argument("--beta-beta", type=float, default=1.5, help="Beta prior beta for GC")
    parser.add_argument(
        "--seed", type=int, default=234338369488123928123758124148124168124258, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader]
    if not rows:
        raise ValueError("Input file contains no records")
    return rows


def derive_contig_name(gene: str) -> str:
    parts = gene.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return gene


def compute_intergenic_distances(rows: List[Dict[str, str]]) -> None:
    by_contig: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        contig = derive_contig_name(row["gene"])
        row["contig"] = contig
        by_contig[contig].append(row)

    for contig_rows in by_contig.values():
        contig_rows.sort(key=lambda r: int(r["start"]))
        for idx, row in enumerate(contig_rows):
            start = int(row["start"])
            end = int(row["end"])
            if idx == 0:
                up = 0
            else:
                prev_end = int(contig_rows[idx - 1]["end"])
                up = start - prev_end - 1
            if idx == len(contig_rows) - 1:
                down = 0
            else:
                next_start = int(contig_rows[idx + 1]["start"])
                down = next_start - end - 1
            row["intergenic_up"] = up
            row["intergenic_down"] = down
            row["overlap_up"] = 1 if up < 0 else 0
            row["overlap_down"] = 1 if down < 0 else 0


def encode_categories(
    rows: List[Dict[str, str]],
    field: str,
    max_unique: int | None = None,
    normalizer=None,
) -> Tuple[List[int], List[str]]:
    if normalizer is None:
        normalizer = lambda x: x
    counter: Counter[str] = Counter()
    normalized: List[str] = []
    for row in rows:
        raw = row.get(field, "") or "UNKNOWN"
        label = normalizer(raw)
        normalized.append(label)
        counter[label] += 1
    replacement: Dict[str, str] = {}
    if max_unique is not None and len(counter) > max_unique:
        keep = {name for name, _ in counter.most_common(max_unique - 1)}
        for name in counter:
            replacement[name] = name if name in keep else "OTHER"
    else:
        for name in counter:
            replacement[name] = name
    encoder = LabelEncoder()
    encoded: List[int] = []
    for row, label in zip(rows, normalized):
        mapped = replacement[label]
        row[f"{field}_label"] = mapped
        code = encoder.encode(mapped)
        encoded.append(code)
        row[f"{field}_encoded"] = code
    return encoded, encoder.labels


def build_records(rows: List[Dict[str, str]]) -> List[PhageRecord]:
    records: List[PhageRecord] = []
    for row in sorted(rows, key=lambda r: (r["contig"], int(r["start"]))):
        length = int(row["length"])
        gc_fraction = float(row["gc_content"])
        gc_successes = int(round(gc_fraction * length))
        gc_successes = max(0, min(length, gc_successes))
        strand_val = row.get("strand", "1").strip()
        strand = 1 if strand_val == "1" else 0
        records.append(
            PhageRecord(
                gene=row["gene"],
                contig=row["contig"],
                start=int(row["start"]),
                end=int(row["end"]),
                length=length,
                strand=strand,
                gc_fraction=gc_fraction,
                gc_successes=gc_successes,
                category=row["category_label"],
                category_encoded=int(row["category_encoded"]),
                rbs=row["rbs_motif_label"],
                rbs_encoded=int(row["rbs_motif_encoded"]),
                intergenic_up=float(row["intergenic_up"]),
                intergenic_down=float(row["intergenic_down"]),
                overlap_up=int(row["overlap_up"]),
                overlap_down=int(row["overlap_down"]),
            )
        )
    return records


def prepare_observations(records: Sequence[PhageRecord]) -> HybridObservations:
    continuous = np.array([[rec.intergenic_up, rec.intergenic_down] for rec in records], dtype=float)
    beta_counts = np.array([[rec.gc_successes, rec.length] for rec in records], dtype=int)
    category = np.array([rec.category_encoded for rec in records], dtype=int)
    rbs = np.array([rec.rbs_encoded for rec in records], dtype=int)
    strand = np.array([rec.strand for rec in records], dtype=int)
    overlap_up = np.array([rec.overlap_up for rec in records], dtype=int)
    overlap_down = np.array([rec.overlap_down for rec in records], dtype=int)
    categorical_arrays = [category, rbs, strand, overlap_up, overlap_down]
    observations = HybridObservations(
        continuous=continuous,
        categorical=categorical_arrays,
        beta_counts=beta_counts,
    )
    return observations


def configure_emission_model(
    continuous: np.ndarray,
    categorical_arrays: Sequence[np.ndarray],
    beta_prior: BetaBinomialPrior,
) -> HybridEmissionModel:
    mu0 = continuous.mean(axis=0)
    centered = continuous - mu0
    scatter = centered.T @ centered
    d = continuous.shape[1]
    psi0 = scatter + np.eye(d) * (1.0 + np.trace(scatter) / max(1, continuous.shape[0]))
    gaussian_prior = GaussianNIWPrior.from_parameters(
        mu0=mu0,
        kappa0=1.0,
        nu0=d + 2.0,
        psi0=psi0,
    )
    categorical_priors = [np.ones(int(arr.max()) + 1, dtype=float) for arr in categorical_arrays]
    return HybridEmissionModel(
        gaussian_prior=gaussian_prior,
        categorical_priors=categorical_priors,
        beta_binomial_prior=beta_prior,
    )


def align_labels(reference: np.ndarray, sample: np.ndarray) -> np.ndarray:
    mapping: Dict[int, int] = {}
    ref_states = int(reference.max()) + 1 if reference.size else 0
    for state in np.unique(sample):
        mask = sample == state
        if not mask.any():
            continue
        counts = np.bincount(reference[mask], minlength=ref_states)
        target = int(np.argmax(counts)) if counts.size else int(state)
        mapping[int(state)] = target
    return np.array([mapping.get(int(s), int(s)) for s in sample], dtype=int)


def compute_confidence(reference: np.ndarray, samples: Sequence[np.ndarray]) -> np.ndarray:
    if not samples:
        return np.ones_like(reference, dtype=float)
    tally = np.zeros_like(reference, dtype=float)
    total = float(len(samples))
    for sample in samples:
        aligned = align_labels(reference, sample)
        tally += (aligned == reference).astype(float)
    return tally / total


def summarize_segments(records: Sequence[PhageRecord], labels: np.ndarray, confidence: np.ndarray) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    if not records:
        return segments
    seg_id = 1
    start_idx = 0
    for idx in range(1, len(records) + 1):
        boundary = False
        if idx == len(records):
            boundary = True
        else:
            same_state = labels[idx] == labels[start_idx]
            same_contig = records[idx].contig == records[start_idx].contig
            boundary = not (same_state and same_contig)
        if boundary:
            block = records[start_idx:idx]
            block_conf = confidence[start_idx:idx]
            avg_conf = float(block_conf.mean()) if block_conf.size else 0.0
            start_gene = block[0]
            end_gene = block[-1]
            dominant_category = Counter(rec.category for rec in block).most_common(1)[0][0]
            segments.append(
                {
                    "segment_id": seg_id,
                    "contig": start_gene.contig,
                    "state": int(labels[start_idx]),
                    "first_gene": start_gene.gene,
                    "last_gene": end_gene.gene,
                    "orf_count": len(block),
                    "start_bp": start_gene.start,
                    "end_bp": end_gene.end,
                    "span_bp": end_gene.end - start_gene.start + 1,
                    "avg_confidence": round(avg_conf, 4),
                    "dominant_category": dominant_category,
                }
            )
            seg_id += 1
            start_idx = idx
    return segments


def write_segments(path: Path, segments: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "segment_id",
        "contig",
        "state",
        "first_gene",
        "last_gene",
        "orf_count",
        "start_bp",
        "end_bp",
        "span_bp",
        "avg_confidence",
        "dominant_category",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for segment in segments:
            writer.writerow(segment)


def main() -> None:
    args = parse_args()
    seed_str = str(args.seed)
    seed_hash_hex = hashlib.sha256(seed_str.encode()).hexdigest()
    seed_int = int(seed_hash_hex, 16) % (2**32 - 1)
    np.random.seed(seed_int)

    rows = load_records(args.input)
    compute_intergenic_distances(rows)
    encode_categories(rows, "category", max_unique=10)
    encode_categories(
        rows, "rbs_motif", normalizer=lambda s: (s or "UNKNOWN").upper()
    )
    records = build_records(rows)
    observations = prepare_observations(records)

    beta_prior = BetaBinomialPrior(alpha0=args.beta_alpha, beta0=args.beta_beta)
    emission_model = configure_emission_model(
        np.asarray(observations.continuous, dtype=float),
        observations.categorical or [],
        beta_prior,
    )

    sampler = DSHDPHMMHybridSampler(
        observations=observations,
        emission_model=emission_model,
        alpha0=args.alpha,
        gamma0=args.gamma,
        rho0=args.rho0,
        rho1=args.rho1,
    )
    sampler.initialize()
    history = sampler.run(num_iterations=args.iterations, burn_in=args.burn_in)
    final_labels = sampler.zt.copy()
    confidence = compute_confidence(final_labels, history.get("zt", []))
    segments = summarize_segments(records, final_labels, confidence)
    write_segments(args.output, segments)


if __name__ == "__main__":
    main()
