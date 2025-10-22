#!/usr/bin/env python3
"""Run MOSA ablation experiments by removing one feature group at a time."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .phage_segmenter import (
    apply_psm_confidence,
    average_adjusted_rand_index,
    build_consensus_segments,
    build_records,
    compute_intergenic_distances,
    compute_psm,
    encode_categories,
    format_log_line,
    hash_seed_to_uint32,
    load_records,
    prepare_observations,
    run_single_sample,
    sample_hyperparameters,
    write_segments,
)
from .sampler import HybridObservations


@dataclass(frozen=True)
class AblationSetting:
    """Metadata describing a single ablation configuration."""

    key: str
    label: str
    removed_feature: str
    description: str


ABLATION_SETTINGS: Tuple[AblationSetting, ...] = (
    AblationSetting(
        key="intergenic",
        label="no_intergenic",
        removed_feature="intergenic_up/down",
        description="Removes upstream and downstream intergenic distances (Gaussian component)",
    ),
    AblationSetting(
        key="length",
        label="no_length",
        removed_feature="length",
        description="Removes ORF length variation from the Beta-Binomial emission",
    ),
    AblationSetting(
        key="gc_successes",
        label="no_gc_successes",
        removed_feature="gc_successes",
        description="Removes GC-content successes while preserving ORF length information",
    ),
    AblationSetting(
        key="category",
        label="no_category",
        removed_feature="category_encoded",
        description="Removes PHROG category assignments from the categorical emissions",
    ),
    AblationSetting(
        key="rbs",
        label="no_rbs",
        removed_feature="rbs_encoded",
        description="Removes ribosome-binding site motif information",
    ),
    AblationSetting(
        key="strand",
        label="no_strand",
        removed_feature="strand",
        description="Removes strand orientation (+/-) categorical observations",
    ),
    AblationSetting(
        key="overlap",
        label="no_overlap",
        removed_feature="overlap_up/down",
        description="Removes overlap indicators with upstream and downstream ORFs",
    ),
)


def _drop_indices(values: Sequence[np.ndarray], indices: Iterable[int]) -> List[np.ndarray]:
    """Return a copy of *values* without the positions in *indices*."""

    drop_set = set(indices)
    return [arr for idx, arr in enumerate(values) if idx not in drop_set]


def create_ablation_observations(
    base_observations: HybridObservations,
    key: str,
) -> HybridObservations:
    """Produce a new observation set with the feature group identified by *key* removed."""

    continuous = (
        None
        if base_observations.continuous is None
        else np.array(base_observations.continuous, copy=True)
    )
    categorical = [
        np.array(arr, copy=True)
        for arr in (base_observations.categorical or [])
    ]
    beta_counts = (
        None
        if base_observations.beta_counts is None
        else np.array(base_observations.beta_counts, copy=True)
    )

    if key == "intergenic":
        continuous = None
    elif key == "length":
        if beta_counts is None:
            raise ValueError("Beta-Binomial observations are required for length ablation")
        successes = beta_counts[:, 0].astype(int, copy=True)
        trials = beta_counts[:, 1].astype(int, copy=True)
        max_trials = int(trials.max()) if trials.size else 1
        if max_trials <= 0:
            max_trials = 1
        constant_trials = np.full_like(successes, max_trials, dtype=int)
        beta_counts = np.column_stack([successes, constant_trials])
    elif key == "gc_successes":
        if beta_counts is None:
            raise ValueError("Beta-Binomial observations are required for GC ablation")
        successes = beta_counts[:, 0].astype(int, copy=True)
        trials = beta_counts[:, 1].astype(int, copy=True)
        total_trials = int(trials.sum())
        if total_trials > 0:
            mean_fraction = float(successes.sum() / total_trials)
        else:
            mean_fraction = 0.0
        adjusted_successes = np.clip(
            np.round(mean_fraction * trials).astype(int),
            0,
            trials,
        )
        beta_counts = np.column_stack([adjusted_successes, trials])
    elif key == "category":
        categorical = _drop_indices(categorical, {0})
    elif key == "rbs":
        categorical = _drop_indices(categorical, {1})
    elif key == "strand":
        categorical = _drop_indices(categorical, {2})
    elif key == "overlap":
        categorical = _drop_indices(categorical, {3, 4})
    else:
        raise ValueError(f"Unknown ablation key: {key}")

    return HybridObservations(
        continuous=continuous,
        categorical=categorical,
        beta_counts=beta_counts,
    )


def _input_stem(input_path: Path) -> str:
    """Mirror the naming logic from ``phage_segmenter`` for output files."""

    stem = input_path.stem
    if stem.endswith("_genes"):
        stem = stem[: -len("_genes")]
    return stem


def _scenario_seed(global_seed: int, key: str) -> int:
    """Deterministically derive a scenario-specific seed from the global seed."""

    combined = int(
        hashlib.sha256(f"{global_seed}:{key}".encode("utf-8")).hexdigest(),
        16,
    )
    return hash_seed_to_uint32(combined)


def run_ablation_experiment(
    setting: AblationSetting,
    records: Sequence,
    base_observations: HybridObservations,
    iterations: int,
    burn_in: int,
    n_samples: int,
    global_seed: int,
    output_root: Path,
    input_stem: str,
) -> dict:
    """Execute a single ablation configuration and persist its outputs."""

    scenario_observations = create_ablation_observations(base_observations, setting.key)
    rng = np.random.default_rng(_scenario_seed(global_seed, setting.key))

    run_results = []
    label_sequences = []
    for run_idx in range(1, n_samples + 1):
        params = sample_hyperparameters(rng)
        result = run_single_sample(
            run_idx=run_idx,
            params=params,
            records=records,
            observations=scenario_observations,
            iterations=iterations,
            burn_in=burn_in,
        )
        run_results.append(result)
        label_sequences.append(result.labels.copy())

    psm = compute_psm(label_sequences, len(records))
    for result in run_results:
        result.mean_confidence = apply_psm_confidence(result.segments, psm)

    consensus_segments = build_consensus_segments(records, psm, threshold=0.5)
    consensus_mean_conf = apply_psm_confidence(consensus_segments, psm)
    consensus_ari = average_adjusted_rand_index(label_sequences)

    ablation_dir = output_root / setting.label
    ablation_dir.mkdir(parents=True, exist_ok=True)

    segments_path = ablation_dir / f"{input_stem}_{setting.label}_segments.tsv"
    log_path = ablation_dir / f"{input_stem}_{setting.label}_segmentation.log"

    write_segments(segments_path, consensus_segments)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"# Removed feature: {setting.removed_feature}\n")
        log_handle.write(f"# Description: {setting.description}\n")
        for result in run_results:
            line = format_log_line(result)
            log_handle.write(line + "\n")
        log_handle.write(
            f"[Consensus] mosaic_num={len(consensus_segments)} "
            f"avg_conf={consensus_mean_conf:.4f} "
            f"ari={consensus_ari:.3f} path={segments_path}\n"
        )

    print(
        f"[Ablation {setting.label}] segments={len(consensus_segments)} "
        f"avg_conf={consensus_mean_conf:.4f} ari={consensus_ari:.3f}"
    )

    return {
        "label": setting.label,
        "removed_feature": setting.removed_feature,
        "description": setting.description,
        "consensus_segments": len(consensus_segments),
        "avg_confidence": consensus_mean_conf,
        "ari": consensus_ari,
        "segments_path": segments_path,
        "log_path": log_path,
    }


def write_summary(output_root: Path, rows: List[dict]) -> Path:
    """Persist a tabular overview of all ablation runs."""

    summary_path = output_root / "ablation_summary.tsv"
    fieldnames = [
        "label",
        "removed_feature",
        "description",
        "consensus_segments",
        "avg_confidence",
        "ari",
        "segments_path",
        "log_path",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "removed_feature": row["removed_feature"],
                    "description": row["description"],
                    "consensus_segments": row["consensus_segments"],
                    "avg_confidence": f"{row['avg_confidence']:.4f}",
                    "ari": f"{row['ari']:.3f}",
                    "segments_path": str(row["segments_path"]),
                    "log_path": str(row["log_path"]),
                }
            )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MOSA ablation experiments by dropping one observation group "
            "(Gaussian, Beta-Binomial, or categorical) per run."
        )
    )
    parser.add_argument("input", type=Path, help="Input TSV with ORF annotations")
    parser.add_argument("output", type=Path, help="Directory for ablation outputs")
    parser.add_argument("--iterations", type=int, default=300, help="Sampler iterations per run")
    parser.add_argument("--burn-in", type=int, default=150, help="Burn-in iterations per run")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of DS-HDP-HMM runs to aggregate for each ablation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=234338369488123928123758124148124168124258,
        help="Global random seed for reproducible ablation experiments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be positive")

    rows = load_records(args.input)
    compute_intergenic_distances(rows)
    encode_categories(rows, "category", max_unique=10)
    encode_categories(rows, "rbs_motif", normalizer=lambda s: (s or "UNKNOWN").upper())
    records = build_records(rows)
    base_observations = prepare_observations(records)

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    stem = _input_stem(args.input)

    summary_rows: List[dict] = []
    for setting in ABLATION_SETTINGS:
        print(
            f"Running ablation '{setting.label}' (remove {setting.removed_feature})"
        )
        summary_rows.append(
            run_ablation_experiment(
                setting=setting,
                records=records,
                base_observations=base_observations,
                iterations=args.iterations,
                burn_in=args.burn_in,
                n_samples=args.n_samples,
                global_seed=args.seed,
                output_root=output_root,
                input_stem=stem,
            )
        )

    summary_path = write_summary(output_root, summary_rows)
    print(f"Wrote ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
