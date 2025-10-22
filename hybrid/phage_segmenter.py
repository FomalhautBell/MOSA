#!/usr/bin/env python3
"""Segment phage genomes using multi-sample DS-HDP-HMM consensus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


@dataclass
class RunParameters:
    """Hyper-parameters sampled for a single DS-HDP-HMM run."""

    seed: int
    alpha: float
    gamma: float
    rho0: float
    rho1: float
    beta_alpha: float
    beta_beta: float


@dataclass
class SegmentSummary:
    """Structured representation of a contiguous phage module."""

    segment_id: int
    contig: str
    state: int
    first_gene: str
    last_gene: str
    orf_count: int
    start_bp: int
    end_bp: int
    span_bp: int
    dominant_category: str
    macro_class: str
    orf_indices: np.ndarray
    avg_confidence: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "contig": self.contig,
            "state": self.state,
            "first_gene": self.first_gene,
            "last_gene": self.last_gene,
            "orf_count": self.orf_count,
            "start_bp": self.start_bp,
            "end_bp": self.end_bp,
            "span_bp": self.span_bp,
            "avg_confidence": round(float(self.avg_confidence), 4),
            "dominant_category": self.dominant_category,
            "macro_class": self.macro_class,
        }


@dataclass
class RunResult:
    """Outputs gathered from one sampling round."""

    index: int
    parameters: RunParameters
    labels: np.ndarray
    segments: List[SegmentSummary]
    runtime_seconds: float
    mean_confidence: float = 0.0


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
        description="Segment phage genomes with bootstrap-style DS-HDP-HMM consensus.",
    )
    parser.add_argument("input", type=Path, help="Input TSV with ORF annotations")
    parser.add_argument("output", type=Path, help="Output path (file or directory) for consensus results")
    parser.add_argument("--iterations", type=int, default=300, help="Number of Gibbs samples per run")
    parser.add_argument("--burn-in", type=int, default=150, help="Burn-in iterations per run")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of randomised sampling runs")
    parser.add_argument(
        "--seed",
        type=int,
        default=234338369488123928123758124148124168124258,
        help="Global seed for hyper-parameter sampling",
    )
    return parser.parse_args()


def _norm_cat(s: str) -> str:
    return (s or "unknown").strip().lower()


DEFAULT_MACRO_MAP: Dict[str, str] = {
    "head and packaging": "Morphogenesis",
    "connector": "Morphogenesis",
    "tail": "Morphogenesis",
    "dna, rna and nucleotide metabolism": "Information processing and lifecycle control",
    "integration and excision": "Information processing and lifecycle control",
    "transcription regulation": "Information processing and lifecycle control",
    "lysis": "Lytic release",
    "moron, auxiliary metabolic gene and host takeover": "Host interaction and auxiliary metabolic functions",
    "other": "Uncharacterized",
    "unknown function": "Uncharacterized",
}


def phrog_to_macro(cat: str, macro_map: Optional[Dict[str, str]] = None) -> str:
    macro_map = macro_map or DEFAULT_MACRO_MAP
    return macro_map.get(_norm_cat(cat), "Uncharacterized")


def sample_hyperparameters(rng: np.random.Generator) -> RunParameters:
    seed = int(rng.integers(0, np.iinfo(np.int64).max))
    alpha = float(rng.uniform(4.0, 10.0))
    gamma = float(rng.uniform(1.0, 4.0))
    rho0 = float(rng.uniform(5.0, 15.0))
    rho1 = float(rng.uniform(0.5, 2.5))
    beta_alpha = float(rng.uniform(1.5, 3.5))
    beta_beta = float(rng.uniform(1.5, 3.5))
    return RunParameters(seed, alpha, gamma, rho0, rho1, beta_alpha, beta_beta)


def hash_seed_to_uint32(seed: int) -> int:
    seed_str = str(seed)
    seed_hash_hex = hashlib.sha256(seed_str.encode()).hexdigest()
    return int(seed_hash_hex, 16) % (2**32 - 1)


def compute_state_weights(reference: np.ndarray, z_samples: Sequence[np.ndarray]) -> np.ndarray:
    N = int(reference.size)
    K = int(reference.max()) + 1 if N else 0
    weights = np.zeros((N, K), dtype=float)
    if not z_samples:
        for i, s in enumerate(reference):
            weights[i, int(s)] = 1.0
        return weights
    for z in z_samples:
        aligned = align_labels(reference, z)
        for i, s in enumerate(aligned):
            if 0 <= s < K:
                weights[i, int(s)] += 1.0
    weights /= float(len(z_samples))
    return weights


def compute_module_macros(
    weights: np.ndarray,
    phrog_labels: Sequence[str],
    macro_map: Optional[Dict[str, str]],
    alpha: float,
    unchar_penalty: float = 0.5,
    modules: Optional[List[List[int]]] = None,
) -> Dict[int, str]:
    N, K = weights.shape
    modules = modules or [[k] for k in range(K)]
    macro_names = [
        "Morphogenesis",
        "Information processing and lifecycle control",
        "Host interaction and auxiliary metabolic functions",
        "Lytic release",
        "Uncharacterized",
    ]
    macro_to_idx = {name: i for i, name in enumerate(macro_names)}
    gene_macro_idx = np.array([macro_to_idx[phrog_to_macro(cat, macro_map)] for cat in phrog_labels], dtype=int)
    state_to_macro: Dict[int, str] = {}
    for mod in modules:
        m = weights[:, mod].sum(axis=1) if len(mod) > 1 else weights[:, mod[0]]
        counts = np.zeros(len(macro_names), dtype=float)
        np.add.at(counts, gene_macro_idx, m)
        counts[macro_to_idx["Uncharacterized"]] *= float(unchar_penalty)
        counts += float(alpha)
        if counts.sum() > 0:
            probs = counts / counts.sum()
        else:
            probs = counts
        winner = macro_names[int(np.argmax(probs))]
        for s in mod:
            state_to_macro[int(s)] = winner
    return state_to_macro


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
    max_unique: Optional[int] = None,
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
    return HybridObservations(
        continuous=continuous,
        categorical=categorical_arrays,
        beta_counts=beta_counts,
    )


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


def build_segment_summaries(records: Sequence[PhageRecord], labels: np.ndarray) -> List[SegmentSummary]:
    segments: List[SegmentSummary] = []
    if not records:
        return segments
    seg_id = 1
    start_idx = 0
    n = len(records)
    while start_idx < n:
        end_idx = start_idx + 1
        while (
            end_idx < n
            and labels[end_idx] == labels[start_idx]
            and records[end_idx].contig == records[start_idx].contig
        ):
            end_idx += 1
        indices = np.arange(start_idx, end_idx, dtype=int)
        block = [records[i] for i in indices]
        dominant_category = Counter(rec.category for rec in block).most_common(1)[0][0]
        segments.append(
            SegmentSummary(
                segment_id=seg_id,
                contig=block[0].contig,
                state=int(labels[start_idx]),
                first_gene=block[0].gene,
                last_gene=block[-1].gene,
                orf_count=len(block),
                start_bp=block[0].start,
                end_bp=block[-1].end,
                span_bp=block[-1].end - block[0].start + 1,
                dominant_category=dominant_category,
                macro_class="Uncharacterized",
                orf_indices=indices,
            )
        )
        seg_id += 1
        start_idx = end_idx
    return segments


def apply_state_annotations(
    segments: List[SegmentSummary],
    records: Sequence[PhageRecord],
    weights: np.ndarray,
    state_to_macro: Dict[int, str],
) -> None:
    for seg in segments:
        if seg.orf_count == 1:
            idx = int(seg.orf_indices[0])
            seg.macro_class = phrog_to_macro(records[idx].category, DEFAULT_MACRO_MAP)
            continue
        state = seg.state
        seg.macro_class = state_to_macro.get(state, "Uncharacterized")
        cat_weights: Dict[str, float] = {}
        for idx in seg.orf_indices:
            category = records[int(idx)].category
            weight = weights[int(idx), state] if state < weights.shape[1] else 0.0
            cat_weights[category] = cat_weights.get(category, 0.0) + weight
        if cat_weights:
            seg.dominant_category = max(cat_weights.items(), key=lambda kv: kv[1])[0]


def compute_psm(label_sequences: Sequence[np.ndarray], length: int) -> np.ndarray:
    if not label_sequences:
        raise ValueError("No label sequences provided for PSM computation")
    psm = np.zeros((length, length), dtype=float)
    for labels in label_sequences:
        for state in np.unique(labels):
            indices = np.where(labels == state)[0]
            if indices.size == 0:
                continue
            psm[np.ix_(indices, indices)] += 1.0
    psm /= float(len(label_sequences))
    np.fill_diagonal(psm, 1.0)
    return psm


def segment_psm_score(psm: np.ndarray, indices: np.ndarray, other_groups: Sequence[np.ndarray]) -> float:
    size = len(indices)
    if size <= 1:
        a_val = 0.0
    else:
        sub = psm[np.ix_(indices, indices)]
        sum_sim = float(sub.sum() - size)  # exclude diagonal ones
        avg_sim = sum_sim / (size * (size - 1))
        a_val = 1.0 - avg_sim
    b_val: Optional[float] = None
    for group in other_groups:
        if len(group) == 0:
            continue
        cross = psm[np.ix_(indices, group)]
        avg_sim = float(cross.mean())
        dist = 1.0 - avg_sim
        if b_val is None or dist < b_val:
            b_val = dist
    if b_val is None:
        b_val = 0.0
    denom = max(a_val, b_val)
    if denom == 0.0:
        return 0.0
    score = (b_val - a_val) / denom
    return float(max(min(score, 1.0), -1.0))


def apply_psm_confidence(segments: List[SegmentSummary], psm: np.ndarray) -> float:
    if not segments:
        return 0.0
    index_groups = [seg.orf_indices for seg in segments]
    scores: List[float] = []
    for idx, seg in enumerate(segments):
        others = index_groups[:idx] + index_groups[idx + 1 :]
        seg.avg_confidence = segment_psm_score(psm, seg.orf_indices, others)
        scores.append(seg.avg_confidence)
    return float(np.mean(scores)) if scores else 0.0


def build_consensus_segments(
    records: Sequence[PhageRecord],
    psm: np.ndarray,
    threshold: float = 0.5,
) -> List[SegmentSummary]:
    segments: List[SegmentSummary] = []
    if not records:
        return segments
    seg_id = 1
    start_idx = 0
    n = len(records)
    while start_idx < n:
        end_idx = start_idx
        while (
            end_idx + 1 < n
            and records[end_idx].contig == records[end_idx + 1].contig
            and psm[end_idx, end_idx + 1] >= threshold
        ):
            end_idx += 1
        indices = np.arange(start_idx, end_idx + 1, dtype=int)
        block = [records[i] for i in indices]
        dominant_category = Counter(rec.category for rec in block).most_common(1)[0][0]
        segments.append(
            SegmentSummary(
                segment_id=seg_id,
                contig=block[0].contig,
                state=seg_id - 1,
                first_gene=block[0].gene,
                last_gene=block[-1].gene,
                orf_count=len(block),
                start_bp=block[0].start,
                end_bp=block[-1].end,
                span_bp=block[-1].end - block[0].start + 1,
                dominant_category=dominant_category,
                macro_class=phrog_to_macro(dominant_category, DEFAULT_MACRO_MAP),
                orf_indices=indices,
            )
        )
        seg_id += 1
        start_idx = end_idx + 1
    return segments


def write_segments(path: Path, segments: Sequence[SegmentSummary]) -> None:
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
        "macro_class",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for segment in segments:
            writer.writerow(segment.to_dict())


def resolve_output_paths(output_target: Path) -> Tuple[Path, Path]:
    if output_target.suffix:
        output_dir = output_target.parent
        final_report = output_target
    else:
        output_dir = output_target
        final_report = output_dir / "consensus_segments.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, final_report


def format_log_line(run: RunResult) -> str:
    params = run.parameters
    return (
        f"[Round {run.index}] seed={params.seed} "
        f"rho0={params.rho0:.2f} rho1={params.rho1:.2f} "
        f"gamma={params.gamma:.2f} alpha={params.alpha:.2f} "
        f"beta_alpha={params.beta_alpha:.2f} beta_beta={params.beta_beta:.2f} "
        f"mosaic_num={len(run.segments)} avg_conf={run.mean_confidence:.4f} "
        f"t={run.runtime_seconds:.3f}s"
    )


def run_single_sample(
    run_idx: int,
    params: RunParameters,
    records: Sequence[PhageRecord],
    observations: HybridObservations,
    iterations: int,
    burn_in: int,
) -> RunResult:
    np.random.seed(hash_seed_to_uint32(params.seed))
    beta_prior = BetaBinomialPrior(alpha0=params.beta_alpha, beta0=params.beta_beta)
    emission_model = configure_emission_model(
        np.asarray(observations.continuous, dtype=float),
        observations.categorical or [],
        beta_prior,
    )
    sampler = DSHDPHMMHybridSampler(
        observations=observations,
        emission_model=emission_model,
        alpha0=params.alpha,
        gamma0=params.gamma,
        rho0=params.rho0,
        rho1=params.rho1,
    )
    sampler.initialize()
    start = time.perf_counter()
    history = sampler.run(num_iterations=iterations, burn_in=burn_in)
    runtime = time.perf_counter() - start
    final_labels = sampler.zt.copy()
    weights = compute_state_weights(final_labels, history.get("zt", []))
    phrog_labels_seq = [rec.category for rec in records]
    state_to_macro = compute_module_macros(
        weights=weights,
        phrog_labels=phrog_labels_seq,
        macro_map=DEFAULT_MACRO_MAP,
        alpha=params.alpha,
        unchar_penalty=0.5,
        modules=None,
    )
    segments = build_segment_summaries(records, final_labels)
    apply_state_annotations(segments, records, weights, state_to_macro)
    return RunResult(
        index=run_idx,
        parameters=params,
        labels=final_labels,
        segments=segments,
        runtime_seconds=runtime,
    )


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")

    rows = load_records(args.input)
    compute_intergenic_distances(rows)
    encode_categories(rows, "category", max_unique=10)
    encode_categories(rows, "rbs_motif", normalizer=lambda s: (s or "UNKNOWN").upper())
    records = build_records(rows)
    observations = prepare_observations(records)

    output_dir, final_report_path = resolve_output_paths(args.output)
    log_path = output_dir / "segmentation.log"

    base_rng = np.random.default_rng(hash_seed_to_uint32(args.seed))
    run_results: List[RunResult] = []
    label_sequences: List[np.ndarray] = []

    for run_idx in range(1, args.n_samples + 1):
        params = sample_hyperparameters(base_rng)
        result = run_single_sample(
            run_idx=run_idx,
            params=params,
            records=records,
            observations=observations,
            iterations=args.iterations,
            burn_in=args.burn_in,
        )
        run_results.append(result)
        label_sequences.append(result.labels.copy())

    psm = compute_psm(label_sequences, len(records))
    for result in run_results:
        result.mean_confidence = apply_psm_confidence(result.segments, psm)

    consensus_segments = build_consensus_segments(records, psm, threshold=0.5)
    consensus_mean_conf = apply_psm_confidence(consensus_segments, psm)

    for result in run_results:
        run_path = output_dir / f"run_{result.index:03d}.tsv"
        write_segments(run_path, result.segments)

    write_segments(final_report_path, consensus_segments)

    with log_path.open("w", encoding="utf-8") as log_handle:
        for result in run_results:
            line = format_log_line(result)
            log_handle.write(line + "\n")
            print(line)
        log_handle.write(
            f"[Consensus] mosaic_num={len(consensus_segments)} avg_conf={consensus_mean_conf:.4f} "
            f"path={final_report_path}\n"
        )
    print(
        f"[Consensus] mosaic_num={len(consensus_segments)} avg_conf={consensus_mean_conf:.4f} "
        f"path={final_report_path}"
    )


if __name__ == "__main__":
    main()
