# MOSA: MOdular Segmentation & Alignment Algorithm 

This repository hosts tooling for working with mixed-type temporal signals and
phage module annotations.  It contains two complementary components:

* **Hybrid DS-HDP-HMM sampler** – a disentangled sticky HDP-HMM implementation
  capable of handling Gaussian, categorical and beta-count observations through
  a unified emission model.  The implementation lives in the
  [`hybrid/`](hybrid) package and exposes the
  `HybridEmissionModel`, `HybridStateStatistics` and
  `DSHDPHMMHybridSampler` classes for probabilistic sequence modelling.
* **Fused Gromov–Wasserstein (FGW) comparison pipeline** – utilities to align
  and compare phage genome module annotations.  The
  [`hybrid.fgw_pipeline`](hybrid/fgw_pipeline.py) module builds the cross-module
  affinity matrix, distance matrices and transport plan used for the FGW
  alignment.  A thin CLI wrapper is provided in
  [`hybrid.compare_fgw`](hybrid/compare_fgw.py).

The repository also ships a small collection of annotated example genomes under
[`phages-example-datas/`](phages-example-datas) that can be used to exercise the
pipeline.

## Requirements

* Python 3.9 or newer
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [POT](https://pythonot.github.io/) (``pip install pot``) for the FGW solver
* [pytest](https://docs.pytest.org/) for running the automated tests (optional)

A virtual environment is highly recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib pot pytest
```

## Repository layout

| Path | Description |
| --- | --- |
| `hybrid/` | Hybrid emission DS-HDP-HMM sampler, FGW comparison utilities and the CLI entry point. |
| `phages-example-datas/` | Example phage module annotations (`*_segments.tsv`) and optional gene annotations (`*_genes.tsv`). |
| `hybrid/tests/` | Pytest-based regression test that exercises the FGW pipeline end-to-end. |

## Running the FGW comparison pipeline

The command line interface wraps `compare_phages_fgw` and materialises all
artefacts (transport plan, affinity matrices, report and plots) in an output
directory.  A minimal example using the bundled data looks as follows:

```bash
python -m hybrid.compare_fgw \
  --a phages-example-datas/GU988610.2/GU988610.2_segments.tsv \
  --b phages-example-datas/IMGVR_UViG_2684623197_000002/IMGVR_UViG_2684623197_000002_segments.tsv \
  --out /tmp/fgw-comparison
```

The command produces the following files inside `/tmp/fgw-comparison`:

* `W.npy`, `T.npy`, `D_A.npy`, `D_B.npy` – NumPy arrays describing the feature
  matrix, optimal transport plan and intra-phage distances.
* `fgwdistance.json`, `p.json`, `q.json` – JSON summaries of the numerical
  outputs.
* `heatmap.png` – a heatmap visualising the affinity matrix and transport plan.
* `report.json` – a structured report containing coverage statistics, runtime
  metadata and the candidate offsets explored during matching.
* `cli_summary.json` – a short JSON blob mirroring the return value of the API.

For programmatic use you can import the API directly:

```python
from hybrid.fgw_pipeline import compare_phages_fgw

result = compare_phages_fgw(
    "phages-example-datas/GU988610.2/GU988610.2_segments.tsv",
    "phages-example-datas/IMGVR_UViG_2684623197_000002/IMGVR_UViG_2684623197_000002_segments.tsv",
    "./outputs",
    method="voting",
    alpha=0.6,
    reg=1e-2,
    max_iter=200,
    tol=1e-6,
)
print(result["fgw_distance"])
```

Refer to the function docstrings in
[`hybrid/fgw_pipeline.py`](hybrid/fgw_pipeline.py) for a detailed description of
all parameters.

## Running the tests

After installing the dependencies you can verify the pipeline end-to-end:

```bash
pytest hybrid/tests/test_fgw_pipeline.py
```

The test downloads no external resources; it reuses the data shipped with the
repository and writes temporary artefacts into a pytest-managed directory.

## Citations

If you use this code or derive ideas from it in academic work, please cite:

* Zhou, D., Gao, Y., Paninski, L. (2020). *Disentangled sticky hierarchical
  Dirichlet process hidden Markov model*. ECML. [arXiv:2004.03019](https://arxiv.org/abs/2004.03019)
* Pastalkova, E., Wang, Y., Mizuseki, K., Buzsáki, G. (2015). *Simultaneous
  extracellular recordings ...* CRCNS.org. DOI: 10.6080/K0KS6PHF28
* Pastalkova, E., Itskov, V., Amarasingham, A., Buzsáki, G. (2008). *Internally
  generated cell assembly sequences in the rat hippocampus*. Science,
  321(5894), 1322–1327.
* Oh, S.M., Rehg, J.M., Balch, T., Dellaert, F. (2008). *Learning and inferring
  motion patterns using parametric segmental switching linear dynamic systems*.
  IJCV, 77(1–3), 103–124.
