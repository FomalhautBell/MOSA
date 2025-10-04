from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as _np_exc:
    raise ImportError(
        "NumPy is required for the FGW comparison pipeline. Install it via "
        "`pip install numpy`."
    ) from _np_exc

try:
    import ot
except ImportError as _pot_exc:
    raise ImportError(
        "The POT library (python-ot) is required to run the FGW alignment. "
        "Install it via `pip install pot` and retry."
    ) from _pot_exc

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DEFAULT_LONG_SEED = (
    "12365812415812364827515812396812414812391824651812434812375812383812356"
)

UNKNOWN_PHROG_LABELS = {
    "other",
    "unknown function",
}

PHROG_10_CANON = {
    "head and packaging",
    "tail",
    "connector",
    "dna, rna and nucleotide metabolism",
    "integration and excision",
    "transcription regulation",
    "lysis",
    "moron, auxiliary metabolic gene and host takeover",
    "other",
    "unknown function",
}

_PHROG_SYNONYM = {
    "head": "head and packaging",
    "dna rna and nucleotide metabolism": "dna, rna and nucleotide metabolism",
    "unknown": "unknown function",
    "moron": "moron, auxiliary metabolic gene and host takeover",
    "auxiliary metabolic gene": "moron, auxiliary metabolic gene and host takeover",
    "host takeover": "moron, auxiliary metabolic gene and host takeover",
}

@dataclass(frozen=True)
class ModuleSegment:
    id: str
    contig: str
    start_bp: float
    end_bp: float
    center_bp: float
    orf_count: int
    span_bp: float
    avg_confidence: float
    macro_class: str
    gene_tags: Tuple[str, ...]

    @property
    def length_bp(self) -> float:
        return float(self.span_bp)


def _norm_tag(x: str) -> str:
    t = str(x).strip().lower().replace("_", " ").replace("-", " ")
    t = " ".join(t.split())
    t = _PHROG_SYNONYM.get(t, t)
    return t if t in PHROG_10_CANON else "other"


def _hash_long_seed(seed_text: str) -> int:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        header = None
        rows: List[Dict[str, str]] = []
        for line_no, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) != len(header):
                raise ValueError(
                    f"Row {line_no} in {path} does not match header length."
                )
            rows.append(dict(zip(header, parts)))
    if not rows:
        raise ValueError(f"Input TSV {path} contained no data rows.")
    return rows


def _load_gene_tags(gene_file: Path) -> Dict[str, List[str]]:
    if not gene_file.exists():
        return {}

    rows = _read_tsv(gene_file)
    required_cols = {"segment_id", "orf_index", "category"}
    header_cols = set(rows[0].keys())
    if not required_cols.issubset(header_cols):
        return {}

    grouped: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for row in rows:
        segment_id = row["segment_id"]
        try:
            orf_idx = int(row["orf_index"])
        except ValueError as exc:  # pragma: no cover - depends on data issues
            raise ValueError(
                f"Invalid ORF index '{row['orf_index']}' in {gene_file}."
            ) from exc
        grouped[segment_id].append((orf_idx, row.get("category", "")))

    ordered: Dict[str, List[str]] = {}
    for segment_id, pairs in grouped.items():
        # Sort by the provided ORF index to respect genomic order.
        sorted_pairs = sorted(pairs, key=lambda item: item[0])
        ordered[segment_id] = [category for _, category in sorted_pairs]
    return ordered


def _load_segments_with_genes(tsv_path: Path) -> List[ModuleSegment]:
    segments_rows = _read_tsv(tsv_path)
    column_aliases = {
        "id": ["id", "segment_id", "module_id"],
        "contig": ["contig", "scaffold", "chromosome"],
        "start_bp": ["start_bp", "start"],
        "end_bp": ["end_bp", "end"],
        "orf_count": ["orf_count", "gene_count"],
        "span_bp": ["span_bp", "length_bp", "length"],
        "avg_confidence": ["avg_confidence", "confidence"],
        "macro_class": ["macro_class", "dominant_category", "macrocategory"],
    }
    available = set(segments_rows[0].keys())
    fmap: Dict[str, str] = {}
    missing = []
    for k, opts in column_aliases.items():
        for c in opts:
            if c in available:
                fmap[k] = c
                break
        else:
            missing.append(k)
    if missing:
        raise ValueError(f"Segments file {tsv_path} missing required columns: {missing}.")
    gene_path = tsv_path.with_name(tsv_path.stem.replace("_segments", "") + "_genes.tsv")
    seq_categories: List[str] | None = None
    gene_map: Dict[str, List[str]] = {}

    if gene_path.exists():
        rows = _read_tsv(gene_path)
        header = set(rows[0].keys())
        if {"segment_id", "orf_index", "category"} <= header:
            grouped: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
            for r in rows:
                try:
                    grouped[r["segment_id"]].append((int(r["orf_index"]), r.get("category", "")))
                except Exception:
                    continue
            for sid, pairs in grouped.items():
                pairs.sort(key=lambda x: x[0])
                gene_map[str(sid)] = [_norm_tag(cat) for _, cat in pairs]
        elif "category" in header:
            seq_categories = [_norm_tag(r["category"]) for r in rows if r.get("category")]
    modules: List[ModuleSegment] = []
    segments_rows.sort(key=lambda r: (r[fmap["contig"]], float(r[fmap["start_bp"]]), float(r[fmap["end"] if "end" in fmap else fmap["end_bp"]])))
    cursor = 0
    total_needed = sum(int(float(r[fmap["orf_count"]])) if r.get(fmap["orf_count"]) not in (None, "", "NA") else 0 for r in segments_rows)
    total_have = len(seq_categories) if seq_categories is not None else None
    if seq_categories is not None and total_have is not None and total_have < total_needed:
        print(f"[WARN] *_genes.tsv 只有 {total_have} 个 category，但需要 {total_needed} 个；不足部分将用宏类补齐。")

    for r in segments_rows:
        start_bp = float(r[fmap["start_bp"]]); end_bp = float(r[fmap["end_bp"]])
        try:
            oc = int(float(r[fmap["orf_count"]]))
        except Exception:
            oc = 0
        macro = _norm_tag(r.get(fmap["macro_class"], "unknown function"))
        seg_id = r[fmap["id"]]

        if seg_id in gene_map:
            tags = gene_map[seg_id]
        elif seq_categories is not None:
            k = max(oc, 0)
            tags = seq_categories[cursor: cursor + k]
            cursor += k
            if len(tags) < k:
                tags += [macro] * (k - len(tags))
        else:
            k = max(oc, 1)
            tags = [macro] * k

        tags = [_norm_tag(t) for t in tags]
        modules.append(ModuleSegment(
            id=seg_id,
            contig=r[fmap["contig"]],
            start_bp=start_bp,
            end_bp=end_bp,
            center_bp=0.5 * (start_bp + end_bp),
            orf_count=oc,
            span_bp=float(r.get(fmap["span_bp"], end_bp - start_bp)),
            avg_confidence=float(r.get(fmap["avg_confidence"], 1.0)),
            macro_class=macro,
            gene_tags=tuple(tags),
        ))
    if sum(len(m.gene_tags) for m in modules) == 0:
        raise RuntimeError("No gene tags were loaded; check *_genes.tsv 'category' column.")

    return modules


def _label_weights(
    tags_a: Sequence[str],
    tags_b: Sequence[str],
    rare_weight: float,
    common_weight: float,
    unknown_alpha: float,
    rare_threshold: int,
) -> Dict[str, float]:
    counter = Counter(tag for tag in tags_a + tags_b if tag)
    weights = {}
    for label, count in counter.items():
        if not label:
            continue
        if label in UNKNOWN_PHROG_LABELS:
            weights[label] = unknown_alpha
        elif count <= rare_threshold:
            weights[label] = rare_weight
        else:
            weights[label] = common_weight
    return weights


def _build_inverted_index(tags: Sequence[str]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(tags):
        if label:
            mapping[label].append(idx)
    return mapping


def _kernel(distance: int, beta: float, bandwidth: int) -> float:
    if abs(distance) > bandwidth:
        return 0.0
    return math.exp(-abs(distance) / beta)


def _vote_candidates(
    tags_a: Sequence[str],
    tags_b: Sequence[str],
    weights: Dict[str, float],
    beta: float,
    bandwidth: int,
    top_k: int,
    cutoff: int,
) -> List[int]:
    pos_a = _build_inverted_index(tags_a)
    pos_b = _build_inverted_index(tags_b)
    votes: Dict[int, float] = defaultdict(float)

    for label in set(pos_a) & set(pos_b):
        positions_a = pos_a[label]
        positions_b = pos_b[label]
        if not positions_a or not positions_b:
            continue
        cartesian = len(positions_a) * len(positions_b)
        weight = weights.get(label, 1.0)
        for i in positions_a:
            for j in positions_b:
                delta = j - i
                kernel = _kernel(abs(j - i), beta, bandwidth)
                if kernel == 0.0:
                    continue
                votes[delta] += weight * kernel

    if not votes:
        return [0]

    sorted_candidates = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    return [offset for offset, _ in sorted_candidates[:top_k]]


def _fft_candidates(
    tags_a: Sequence[str],
    tags_b: Sequence[str],
    top_k: int,
    fallback: Iterable[int],
    weights: Dict[str, float],
) -> List[int]:
    if scipy_signal is None:
        return list(fallback)

    m = len(tags_a)
    n = len(tags_b)
    if m == 0 or n == 0:
        return [0]

    label_set = sorted({tag for tag in tags_a + tags_b if tag})
    if not label_set:
        return [0]

    correlations = None
    for label in label_set:
        weight = weights.get(label, 1.0)
        vec_a = np.zeros(m)
        vec_b = np.zeros(n)
        for idx, tag in enumerate(tags_a):
            if tag == label:
                vec_a[idx] = weight
        for idx, tag in enumerate(tags_b):
            if tag == label:
                vec_b[idx] = weight
        corr = scipy_signal.fftconvolve(vec_a, vec_b[::-1], mode="full")
        if correlations is None:
            correlations = corr
        else:
            correlations += corr

    assert correlations is not None  # for type checking
    offsets = np.arange(-(n - 1), m)
    top_indices = np.argsort(correlations)[::-1][:top_k]
    ranked: List[Tuple[float, int]] = []
    seen: set[int] = set()
    for idx in top_indices:
        offset = int(offsets[idx])
        if offset in seen:
            continue
        seen.add(offset)
        ranked.append((float(correlations[idx]), offset))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [offset for _, offset in ranked] or [0]


def _band_score(
    tags_a: Sequence[str],
    tags_b: Sequence[str],
    offset: int,
    beta: float,
    bandwidth: int,
    unknown_alpha: float,
) -> float:
    if not tags_a or not tags_b:
        return 0.0

    matches = []
    unique_js: set[int] = set()
    total_score = 0.0
    for i, label_a in enumerate(tags_a):
        j_center = i + offset
        j_min = max(0, j_center - bandwidth)
        j_max = min(len(tags_b) - 1, j_center + bandwidth)
        if j_min > j_max:
            continue
        for j in range(j_min, j_max + 1):
            label_b = tags_b[j]
            if label_a != label_b:
                continue
            penalty = math.exp(-abs((j - i) - offset) / beta)
            if label_a in UNKNOWN_PHROG_LABELS:
                match_score = unknown_alpha * penalty
            else:
                match_score = penalty
            total_score += match_score
            matches.append(match_score)
            unique_js.add(j)

    if not matches:
        return 0.0

    matched_pairs = len(matches)
    coverage = matched_pairs / max(1.0, math.sqrt(len(tags_a) * max(len(unique_js), 1)))
    omega = total_score / max(len(tags_a), len(tags_b))
    position = math.exp(-abs(offset) / beta)
    gamma = 0.7
    score = (omega ** gamma) * (coverage ** ((1 - gamma) / 2)) * (position ** ((1 - gamma) / 2))
    return float(max(0.0, min(score, 1.0)))


def _compute_w_entry(
    tags_a: Sequence[str],
    tags_b: Sequence[str],
    method: str,
    params: Dict[str, float],
    report_candidates: Dict[Tuple[int, int], List[int]],
    key: Tuple[int, int],
) -> float:
    beta = float(params.get("beta", 4.0))
    bandwidth = int(params.get("bandwidth", 12))
    top_k = int(params.get("top_k", 3))
    cutoff = int(params.get("cutoff", 800))
    unknown_alpha = float(params.get("unknown_alpha", 0.3))
    rare_weight = float(params.get("rare_weight", 1.5))
    common_weight = float(params.get("common_weight", 1.0))
    rare_threshold = int(params.get("rare_threshold", 2))

    weights = _label_weights(tags_a, tags_b, rare_weight, common_weight, unknown_alpha, rare_threshold)
    if method == "fft":
        fallback_offsets = _vote_candidates(tags_a, tags_b, weights, beta, bandwidth, top_k, cutoff)
        offsets = _fft_candidates(tags_a, tags_b, top_k, fallback_offsets, weights)
    else:
        offsets = _vote_candidates(tags_a, tags_b, weights, beta, bandwidth, top_k, cutoff)

    report_candidates[key] = offsets
    best_score = 0.0
    for offset in offsets:
        score = _band_score(tags_a, tags_b, offset, beta, bandwidth, unknown_alpha)
        best_score = max(best_score, score)
    return best_score


def _pre_filter(
    module_a: ModuleSegment,
    module_b: ModuleSegment,
    quality: str,
    length_ratio_tol: float,
    jaccard_threshold: float,
) -> bool:
    len_a = getattr(module_a, quality)
    len_b = getattr(module_b, quality)
    if len_b == 0 or len_a == 0:
        return True
    ratio = max(len_a, len_b) / max(1e-6, min(len_a, len_b))
    if ratio > length_ratio_tol:
        return False

    set_a = {tag for tag in module_a.gene_tags if tag}
    set_b = {tag for tag in module_b.gene_tags if tag}
    if not set_a or not set_b:
        return True
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return True
    jaccard = inter / union
    return jaccard >= jaccard_threshold


def compute_feature_matrix(
    modules_a: Sequence[ModuleSegment],
    modules_b: Sequence[ModuleSegment],
    method: str,
    params: Dict[str, float],
    pre_filter: bool = True,
    quality_field: str = "span_bp",
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[int]]]:
    """Compute the inter-module affinity/feature matrix ``W``."""

    n_a = len(modules_a)
    n_b = len(modules_b)
    W = np.zeros((n_a, n_b), dtype=float)
    candidates: Dict[Tuple[int, int], List[int]] = {}

    length_ratio_tol = float(params.get("length_ratio_tol", 5.0))
    jaccard_threshold = float(params.get("jaccard_threshold", 0.05))

    for i, module_a in enumerate(modules_a):
        for j, module_b in enumerate(modules_b):
            if pre_filter:
                if not _pre_filter(
                    module_a,
                    module_b,
                    quality=quality_field,
                    length_ratio_tol=length_ratio_tol,
                    jaccard_threshold=jaccard_threshold,
                ):
                    W[i, j] = 0.0
                    candidates[(i, j)] = []
                    continue
            score = _compute_w_entry(
                module_a.gene_tags,
                module_b.gene_tags,
                method=method,
                params=params,
                report_candidates=candidates,
                key=(i, j),
            )
            W[i, j] = score
    return W, candidates


def compute_distance_matrix(modules: Sequence[ModuleSegment], high_const: float = 1.5) -> np.ndarray:
    """Construct the intra-phage distance matrix based on module centres."""

    n = len(modules)
    D = np.zeros((n, n), dtype=float)
    contig_lengths: Dict[str, float] = defaultdict(float)
    for module in modules:
        contig_lengths[module.contig] = max(contig_lengths[module.contig], module.end_bp)

    for i, mod_i in enumerate(modules):
        for j, mod_j in enumerate(modules):
            if mod_i.contig != mod_j.contig:
                D[i, j] = high_const
                continue
            contig_length = contig_lengths[mod_i.contig]
            if contig_length <= 0:
                D[i, j] = 0.0
                continue
            D[i, j] = abs(mod_i.center_bp - mod_j.center_bp) / contig_length
    return D


def _quality_vector(modules: Sequence[ModuleSegment], field: str) -> np.ndarray:
    values = np.array([getattr(module, field) for module in modules], dtype=float)
    total = float(values.sum())
    if total <= 0:
        # Fall back to ORF count if span based quality is not informative.
        values = np.array([float(module.orf_count) for module in modules], dtype=float)
        total = float(values.sum())
    if total <= 0:
        # Uniform distribution as a last resort.
        values = np.ones(len(modules), dtype=float)
        total = float(values.sum())
    return values / total


def _fused_gromov_wasserstein(
    C: np.ndarray,
    D_a: np.ndarray,
    D_b: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    alpha: float,
    reg: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, float]:
    """Run the FGW solver using POT's implementation."""

    try:
        solver = ot.gromov.fused_gromov_wasserstein
        use_entropic = False
    except AttributeError:  # pragma: no cover - older POT versions
        solver = ot.gromov.entropic_gromov_wasserstein
        use_entropic = True

    if use_entropic:
        T, log = solver(
            C,
            D_a,
            D_b,
            p,
            q,
            alpha=alpha,
            epsilon=reg,
            log=True,
            max_iter=max_iter,
            tol=tol,
            verbose=False,
        )
    else:
        try:
            T, log = solver(
                C,
                D_a,
                D_b,
                p,
                q,
                alpha=alpha,
                armijo=False,
                log=True,
                verbose=False,
                max_iter=max_iter,
                tol=tol,
            )
        except TypeError:  # pragma: no cover - fallback for API changes
            T, log = solver(
                C,
                D_a,
                D_b,
                p,
                q,
                alpha=alpha,
                log=True,
                max_iter=max_iter,
                tol=tol,
            )
    fgw_dist = float(log.get("fgw_dist", log.get("gw_dist", np.nan)))
    return T, fgw_dist


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def _plot_heatmaps(
    T: np.ndarray,
    W: np.ndarray,
    modules_a: Sequence,
    modules_b: Sequence,
    out_file: Path,
    colors: tuple[str, str, str] = ("#8EDEB6", "#E7F5FF", "#BA99F5"),
    border_color: str = "#BEBEBE",
) -> None:
    """Create heatmaps for T and element-wise T ⊙ W with a 3-color gradient and gray borders."""
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    matrices = [T, T * W]
    titles = ["Transport plan T", "Element-wise product T ⊙ W"]

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
        num_y, num_x = matrix.shape
        for y in np.arange(-0.5, num_y, 1):
            ax.axhline(y=y, color=border_color, linewidth=0.5)
        for x in np.arange(-0.5, num_x, 1):
            ax.axvline(x=x, color=border_color, linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Modules of B")
        ax.set_ylabel("Modules of A")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def compare_phages_fgw(
    tsv_a: str,
    tsv_b: str,
    out_dir: str,
    *,
    method: str = "voting",
    alpha: float = 0.6,
    reg: float = 1e-2,
    max_iter: int = 1000,
    tol: float = 1e-6,
    quality: str = "span_bp",
    pre_filter: bool = True,
    seed: str = DEFAULT_LONG_SEED,
    match_threshold: float = 0.05,
    high_const: float = 1.5,
    **kwargs,
) -> Dict[str, float]:
    """Execute the full FGW comparison pipeline.

    Parameters
    ----------
    tsv_a, tsv_b:
        Paths to the ``*_segments.tsv`` files for the two phages.
    out_dir:
        Directory where all outputs (matrices, reports, figures) will be stored.
    method:
        Either ``"voting"`` or ``"fft"``; controls the candidate generation
        strategy when computing the cross-module affinity matrix ``W``.
    alpha, reg, max_iter, tol:
        Parameters forwarded to the FGW solver.
    quality:
        Field used to build the marginal distributions (``span_bp`` or
        ``orf_count``).
    pre_filter:
        Whether to apply the inexpensive heuristics prior to expensive scoring.
    seed:
        Either an integer or an arbitrary string controlling the NumPy RNG.
    match_threshold:
        Threshold applied to the transport plan when summarising coverage.
    high_const:
        Distance penalty for cross-contig pairs.
    kwargs:
        Additional parameters forwarded to :func:`compute_feature_matrix`.
    """

    if method not in {"voting", "fft"}:
        raise ValueError("method must be either 'voting' or 'fft'")

    start_time = time.time()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng_seed = _hash_long_seed(str(seed))
    np.random.default_rng(rng_seed)


def _resolve_and_load_segments(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file():
        folder = p.parent
        seg_name = p.name
        stem = p.stem
        if stem.endswith("_segments"):
            base = stem[: -len("_segments")]
        else:
            base = stem
        candidates = [
            f"{base}_genes.tsv",
            f"{stem}_genes.tsv",
            f"{base}_genes.csv",
            f"{stem}_genes.csv",
            "genes.tsv",
            "genes.csv",
        ]
        found_gene = None
        for cand in candidates:
            if (folder / cand).exists():
                found_gene = cand
                break
        genes_name = found_gene if found_gene is not None else candidates[0]
        segments_name = seg_name
        return _load_segments_with_genes(folder, segments_name, genes_name)
    else:
        folder = p
        seg_files = list(folder.glob("*_segments.tsv")) + list(folder.glob("segments.tsv"))
        if seg_files:
            seg_path = seg_files[0]
            stem = seg_path.stem
            if stem.endswith("_segments"):
                base = stem[: -len("_segments")]
            else:
                base = stem
            candidates = [
                f"{base}_genes.tsv",
                f"{stem}_genes.tsv",
                "genes.tsv",
            ]
            found_gene = next((c for c in candidates if (folder / c).exists()), candidates[0])
            return _load_segments_with_genes(folder, seg_path.name, found_gene)
        else:
            raise FileNotFoundError(f"No segments file found in directory {folder}. Expected '*_segments.tsv' or 'segments.tsv'.")
    modules_a = _resolve_and_load_segments(tsv_a)
    modules_b = _resolve_and_load_segments(tsv_b)

    W, candidate_offsets = compute_feature_matrix(
        modules_a,
        modules_b,
        method=method,
        params=kwargs,
        pre_filter=pre_filter,
        quality_field=quality,
    )

    D_a = compute_distance_matrix(modules_a, high_const=high_const)
    D_b = compute_distance_matrix(modules_b, high_const=high_const)

    p = _quality_vector(modules_a, quality)
    q = _quality_vector(modules_b, quality)

    C = 1.0 - W
    T, fgw_distance = _fused_gromov_wasserstein(
        C,
        D_a,
        D_b,
        p,
        q,
        alpha=float(alpha),
        reg=float(reg),
        max_iter=int(max_iter),
        tol=float(tol),
    )

    duration = time.time() - start_time

    # Persist artefacts
    np.save(out_path / "W.npy", W)
    np.save(out_path / "D_A.npy", D_a)
    np.save(out_path / "D_B.npy", D_b)
    np.save(out_path / "T.npy", T)
    _save_json(out_path / "p.json", {"p": p.tolist()})
    _save_json(out_path / "q.json", {"q": q.tolist()})
    _save_json(
        out_path / "fgwdistance.json",
        {
            "fgw_distance": fgw_distance,
            "alpha": float(alpha),
            "p_sum": float(p.sum()),
            "q_sum": float(q.sum()),
            "W_shape": list(W.shape),
            "D_shapes": [list(D_a.shape), list(D_b.shape)],
        },
    )
    _plot_heatmaps(T, W, modules_a, modules_b, out_path / "heatmap.png")

    cov_a = float(np.sum(p[np.sum(T, axis=1) >= match_threshold]))
    cov_b = float(np.sum(q[np.sum(T, axis=0) >= match_threshold]))
    matched_pairs = int(np.sum(T > match_threshold))
    W_min, W_max, W_mean = float(W.min()), float(W.max()), float(W.mean())
    sum_TW = float(np.sum(T * W))
    norm_base = float(min(len(modules_a), len(modules_b))) if min(len(modules_a), len(modules_b)) > 0 else 1.0
    score_norm = sum_TW / norm_base

    report_payload = {
        "fgw_distance": fgw_distance,
        "cov_A_to_B": cov_a,
        "cov_B_to_A": cov_b,
        "matched_pairs": matched_pairs,
        "method": method,
        "pre_filter": bool(pre_filter),
        "match_threshold": match_threshold,
        "alpha": alpha,
        "reg": reg,
        "max_iter": max_iter,
        "tol": tol,
        "quality": quality,
        "duration_seconds": duration,
        "seed": rng_seed,
        "candidate_offsets": {
            f"{i}-{j}": offsets for (i, j), offsets in candidate_offsets.items()
        },
        "num_modules": {
            "A": len(modules_a),
            "B": len(modules_b),
        },
        "W_min": W_min,
        "W_max": W_max,
        "W_mean": W_mean,
        "sum_TW": sum_TW,
        "score_norm": score_norm,
        "timestamp": time.time(),
    }
    _save_json(out_path / "report.json", report_payload)

    return {"fgw_distance": fgw_distance, "duration_seconds": duration}


__all__ = ["compare_phages_fgw", "compute_feature_matrix", "compute_distance_matrix"]