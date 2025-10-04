from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid.fgw_pipeline import compare_phages_fgw


DATA_DIR = Path("phages-example-datas")


def test_compare_phages_fgw(tmp_path: Path) -> None:
    tsv_a = DATA_DIR / "GU988610.2" / "GU988610.2_segments.tsv"
    tsv_b = DATA_DIR / "IMGVR_UViG_2684623197_000002" / "IMGVR_UViG_2684623197_000002_segments.tsv"

    result = compare_phages_fgw(
        str(tsv_a),
        str(tsv_b),
        str(tmp_path),
        method="voting",
        alpha=0.6,
        reg=1e-1,
        max_iter=100,
        tol=1e-5,
        match_threshold=0.02,
    )

    assert "fgw_distance" in result
    assert (tmp_path / "W.npy").exists()
    assert (tmp_path / "T.npy").exists()
    assert (tmp_path / "heatmap.png").exists()

    W = np.load(tmp_path / "W.npy")
    T = np.load(tmp_path / "T.npy")
    assert W.shape[0] > 0 and W.shape[1] > 0
    assert T.shape == W.shape

    with (tmp_path / "report.json").open("r", encoding="utf-8") as fh:
        report = json.load(fh)
    assert report["num_modules"]["A"] == W.shape[0]
    assert report["num_modules"]["B"] == W.shape[1]
