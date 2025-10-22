import pytest

np = pytest.importorskip("numpy")

from hybrid.ablation import ABLATION_SETTINGS, create_ablation_observations
from hybrid.phage_segmenter import configure_emission_model
from hybrid.sampler import HybridObservations
from hybrid.emissions import BetaBinomialPrior


def _make_base_observations() -> HybridObservations:
    continuous = np.array(
        [[2.0, 5.0], [3.0, 7.0], [4.0, 9.0]],
        dtype=float,
    )
    categorical = [
        np.array([0, 1, 2], dtype=int),
        np.array([1, 0, 1], dtype=int),
        np.array([0, 1, 0], dtype=int),
        np.array([1, 0, 1], dtype=int),
        np.array([0, 1, 0], dtype=int),
    ]
    beta_counts = np.array([[6, 12], [9, 18], [12, 24]], dtype=int)
    return HybridObservations(continuous=continuous, categorical=categorical, beta_counts=beta_counts)


def test_create_ablation_intergenic_removes_continuous_component():
    base = _make_base_observations()
    ablated = create_ablation_observations(base, "intergenic")
    assert ablated.continuous is None
    assert ablated.beta_counts is not None
    assert ablated.categorical is not None
    assert len(ablated.categorical) == 5


def test_create_ablation_length_sets_constant_trials():
    base = _make_base_observations()
    ablated = create_ablation_observations(base, "length")
    np.testing.assert_array_equal(ablated.beta_counts[:, 0], base.beta_counts[:, 0])
    assert np.all(ablated.beta_counts[:, 1] == base.beta_counts[:, 1].max())
    # Ensure the original observations are untouched
    np.testing.assert_array_equal(base.beta_counts[:, 1], np.array([12, 18, 24]))


def test_create_ablation_gc_successes_preserves_lengths():
    base = _make_base_observations()
    ablated = create_ablation_observations(base, "gc_successes")
    np.testing.assert_array_equal(ablated.beta_counts[:, 1], base.beta_counts[:, 1])
    total_success = base.beta_counts[:, 0].sum()
    total_trials = base.beta_counts[:, 1].sum()
    mean_fraction = total_success / total_trials
    expected_success = np.clip(
        np.round(mean_fraction * base.beta_counts[:, 1]).astype(int),
        0,
        base.beta_counts[:, 1],
    )
    np.testing.assert_array_equal(ablated.beta_counts[:, 0], expected_success)


def test_create_ablation_removes_expected_categorical_indices():
    base = _make_base_observations()
    # Remove PHROG categories
    no_cat = create_ablation_observations(base, "category")
    assert len(no_cat.categorical) == 4
    np.testing.assert_array_equal(no_cat.categorical[0], base.categorical[1])
    # Remove RBS motifs
    no_rbs = create_ablation_observations(base, "rbs")
    assert len(no_rbs.categorical) == 4
    np.testing.assert_array_equal(no_rbs.categorical[1], base.categorical[2])
    # Remove strand information
    no_strand = create_ablation_observations(base, "strand")
    assert len(no_strand.categorical) == 4
    np.testing.assert_array_equal(no_strand.categorical[2], base.categorical[3])
    # Remove overlaps
    no_overlap = create_ablation_observations(base, "overlap")
    assert len(no_overlap.categorical) == 3
    for idx in range(3):
        np.testing.assert_array_equal(no_overlap.categorical[idx], base.categorical[idx])


def test_configure_emission_model_handles_optional_inputs():
    base = _make_base_observations()
    full_model = configure_emission_model(
        base.continuous,
        base.categorical or [],
        BetaBinomialPrior(alpha0=2.0, beta0=3.0),
    )
    assert full_model.has_continuous
    assert full_model.categorical_priors
    assert full_model.has_beta_binomial

    stripped_model = configure_emission_model(None, [], None)
    assert not stripped_model.has_continuous
    assert not stripped_model.categorical_priors
    assert not stripped_model.has_beta_binomial


def test_ablation_settings_cover_seven_unique_keys():
    keys = {setting.key for setting in ABLATION_SETTINGS}
    assert len(keys) == 7
