"""Command line interface for the phage FGW comparison pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid.fgw_pipeline import compare_phages_fgw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two phage genomes using a fused Gromov–Wasserstein pipeline.",
    )
    parser.add_argument("--a", dest="tsv_a", required=True, help="Path to the first *_segments.tsv file")
    parser.add_argument("--b", dest="tsv_b", required=True, help="Path to the second *_segments.tsv file")
    parser.add_argument("--out", dest="out_dir", required=True, help="Output directory for all artefacts")
    parser.add_argument("--method", choices=["voting", "fft"], default="voting", help="Candidate generation method")
    parser.add_argument("--alpha", type=float, default=0.6, help="FGW structural vs feature trade-off")
    parser.add_argument("--reg", type=float, default=1e-2, help="Entropy regularisation strength")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum FGW iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="FGW convergence tolerance")
    parser.add_argument("--quality", choices=["span_bp", "orf_count"], default="span_bp", help="Mass/quality field")
    parser.add_argument("--no-pre-filter", action="store_true", help="Disable inexpensive heuristic pre-filtering")
    parser.add_argument("--seed", default=12365812415812364827515812396812414812391824651812434812375812383812356, help="Random seed string (defaults to specification seed)")
    parser.add_argument("--match-threshold", type=float, default=0.05, help="Threshold when summarising T coverage")
    parser.add_argument("--high-const", type=float, default=1.5, help="Distance assigned across contigs")
    parser.add_argument("--unknown-alpha", type=float, default=0.07, help="Score used for unknown/other labels")
    parser.add_argument("--beta", type=float, default=4.0, help="Exponential decay factor for offsets")
    parser.add_argument("--bandwidth", type=int, default=12, help="Offset bandwidth used during refinement")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate offsets examined")
    parser.add_argument("--cutoff", type=int, default=800, help="Cartesian product cut-off for voting")
    parser.add_argument("--rare-weight", type=float, default=1.5, help="Weight for rare PHROG labels")
    parser.add_argument("--common-weight", type=float, default=1.0, help="Weight for common PHROG labels")
    parser.add_argument("--rare-threshold", type=int, default=2, help="Frequency threshold to classify as rare")
    parser.add_argument("--length-ratio-tol", type=float, default=5.0, help="Maximum allowed module length ratio")
    parser.add_argument("--jaccard-threshold", type=float, default=0.05, help="Minimum bag-of-words Jaccard overlap")
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    params = dict(
        method=parsed.method,
        alpha=parsed.alpha,
        reg=parsed.reg,
        max_iter=parsed.max_iter,
        tol=parsed.tol,
        quality=parsed.quality,
        pre_filter=not parsed.no_pre_filter,
        seed=parsed.seed if parsed.seed is not None else None,
        match_threshold=parsed.match_threshold,
        high_const=parsed.high_const,
        unknown_alpha=parsed.unknown_alpha,
        beta=parsed.beta,
        bandwidth=parsed.bandwidth,
        top_k=parsed.top_k,
        cutoff=parsed.cutoff,
        rare_weight=parsed.rare_weight,
        common_weight=parsed.common_weight,
        rare_threshold=parsed.rare_threshold,
        length_ratio_tol=parsed.length_ratio_tol,
        jaccard_threshold=parsed.jaccard_threshold,
    )

    # Remove None seeds so the default specification seed is used downstream.
    if params["seed"] is None:
        params.pop("seed")

    result = compare_phages_fgw(
        parsed.tsv_a,
        parsed.tsv_b,
        parsed.out_dir,
        **params,
    )

    output_path = Path(parsed.out_dir) / "cli_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution only
    raise SystemExit(main())