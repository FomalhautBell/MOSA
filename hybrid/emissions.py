"""Emission models for hybrid continuous / categorical observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.special as ssp


ArrayLike = Sequence[float]


def _unpack_observation(
    observation: Union[
        Tuple[Optional[np.ndarray], Optional[Sequence[int]]],
        Tuple[Optional[np.ndarray], Optional[Sequence[int]], Optional[Tuple[int, int]]],
    ]
) -> Tuple[
    Optional[np.ndarray],
    Optional[Sequence[int]],
    Optional[Tuple[int, int]],
]:
    """Return observation components allowing legacy (cont, cats) tuples."""

    if len(observation) == 2:
        cont, cats = observation
        beta_counts = None
    elif len(observation) == 3:
        cont, cats, beta_counts = observation
    else:
        raise ValueError("Observation must contain 2 or 3 elements")
    return cont, cats, beta_counts


def _ensure_array(name: str, value: Optional[np.ndarray]) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} must be provided for the Gaussian prior")
    return np.asarray(value, dtype=float)


@dataclass
class GaussianNIWPrior:
    """Normal-Inverse-Wishart prior hyper-parameters."""

    mu0: np.ndarray
    kappa0: float
    nu0: float
    psi0: np.ndarray

    @classmethod
    def from_parameters(
        cls,
        mu0: ArrayLike,
        kappa0: float,
        nu0: float,
        psi0: ArrayLike,
    ) -> "GaussianNIWPrior":
        mu0_arr = _ensure_array("mu0", np.asarray(mu0, dtype=float))
        psi0_arr = _ensure_array("psi0", np.asarray(psi0, dtype=float))
        if psi0_arr.shape[0] != psi0_arr.shape[1]:
            raise ValueError("psi0 must be a square matrix")
        if mu0_arr.shape[0] != psi0_arr.shape[0]:
            raise ValueError("mu0 and psi0 must have matching dimensionality")
        if nu0 <= mu0_arr.shape[0] - 1:
            raise ValueError(
                "nu0 must be greater than the dimensionality minus one for a valid NIW prior"
            )
        return cls(mu0=mu0_arr, kappa0=float(kappa0), nu0=float(nu0), psi0=psi0_arr)


@dataclass
class BetaBinomialPrior:
    """Beta prior for the Beta-Binomial emission component."""

    alpha0: float
    beta0: float

    def __post_init__(self) -> None:
        if self.alpha0 <= 0 or self.beta0 <= 0:
            raise ValueError("Beta prior parameters must be strictly positive")



class HybridStateStatistics:
    """Sufficient statistics for a single hidden state."""

    def __init__(self, emission_model: "HybridEmissionModel") -> None:
        self._model = emission_model
        self.count: int = 0
        if emission_model.has_continuous:
            d = emission_model.continuous_dim
            self.sum_cont = np.zeros(d, dtype=float)
            self.sum_outer = np.zeros((d, d), dtype=float)
        else:
            self.sum_cont = np.array([], dtype=float)
            self.sum_outer = np.zeros((0, 0), dtype=float)
        self.cat_counts: List[np.ndarray] = [
            np.zeros_like(prior, dtype=float) for prior in emission_model.categorical_priors
        ]
        if emission_model.has_beta_binomial:
            self.successes = 0.0
            self.trials = 0.0
        else:
            self.successes = 0.0
            self.trials = 0.0


    def add_observation(
        self,
        observation: Union[
            Tuple[Optional[np.ndarray], Optional[Sequence[int]]],
            Tuple[Optional[np.ndarray], Optional[Sequence[int]], Optional[Tuple[int, int]]],
        ],
    ) -> None:
        cont, cats, beta_counts = _unpack_observation(observation)
        if cont is not None:
            cont = np.asarray(cont, dtype=float)
            self.sum_cont += cont
            self.sum_outer += np.outer(cont, cont)
        if cats is not None:
            for idx, cat in enumerate(cats):
                self.cat_counts[idx][cat] += 1.0
        if beta_counts is not None:
            succ, trials = beta_counts
            self.successes += float(succ)
            self.trials += float(trials)
        self.count += 1

    def remove_observation(
        self,
        observation: Union[
            Tuple[Optional[np.ndarray], Optional[Sequence[int]]],
            Tuple[Optional[np.ndarray], Optional[Sequence[int]], Optional[Tuple[int, int]]],
        ],
    ) -> None:
        cont, cats, beta_counts = _unpack_observation(observation)
        if cont is not None:
            cont = np.asarray(cont, dtype=float)
            self.sum_cont -= cont
            self.sum_outer -= np.outer(cont, cont)
        if cats is not None:
            for idx, cat in enumerate(cats):
                self.cat_counts[idx][cat] -= 1.0
        if beta_counts is not None:
            succ, trials = beta_counts
            self.successes -= float(succ)
            self.trials -= float(trials)
        self.count -= 1

    def predictive_log_prob(
        self,
        observation: Union[
            Tuple[Optional[np.ndarray], Optional[Sequence[int]]],
            Tuple[Optional[np.ndarray], Optional[Sequence[int]], Optional[Tuple[int, int]]],
        ],
    ) -> float:
        return self._model.predictive_log_prob(self, observation)


class HybridEmissionModel:
    """Hybrid emission model with Gaussian and categorical components."""

    def __init__(
        self,
        gaussian_prior: Optional[GaussianNIWPrior] = None,
        categorical_priors: Optional[Iterable[ArrayLike]] = None,
        beta_binomial_prior: Optional[BetaBinomialPrior] = None,
    ) -> None:
        self.gaussian_prior = gaussian_prior
        self.categorical_priors: List[np.ndarray] = [
            np.asarray(prior, dtype=float) for prior in (categorical_priors or [])
        ]
        for prior in self.categorical_priors:
            if prior.ndim != 1:
                raise ValueError("Categorical priors must be one-dimensional vectors")
            if np.any(prior <= 0):
                raise ValueError("Categorical priors must have strictly positive entries")
        self.beta_binomial_prior = beta_binomial_prior

    @property
    def has_continuous(self) -> bool:
        return self.gaussian_prior is not None

    @property
    def continuous_dim(self) -> int:
        if self.gaussian_prior is None:
            return 0
        return int(self.gaussian_prior.mu0.shape[0])

    def create_state(self) -> HybridStateStatistics:
        return HybridStateStatistics(self)

    def predictive_log_prob(
        self,
        state: HybridStateStatistics,
        observation: Union[
            Tuple[Optional[np.ndarray], Optional[Sequence[int]]],
            Tuple[Optional[np.ndarray], Optional[Sequence[int]], Optional[Tuple[int, int]]],
        ],
    ) -> float:
        cont, cats, beta_counts = _unpack_observation(observation)
        log_prob = 0.0
        if self.has_continuous:
            if cont is None:
                raise ValueError("Continuous observation expected but not provided")
            log_prob += self._gaussian_log_predictive(state, np.asarray(cont, dtype=float))
        if self.categorical_priors and cats is None:
            raise ValueError("Categorical observation expected but not provided")
        if self.categorical_priors:
            log_prob += self._categorical_log_predictive(state, cats)
        if self.has_beta_binomial:
            if beta_counts is None:
                raise ValueError("Beta-Binomial observation expected but not provided")
            log_prob += self._beta_binomial_log_predictive(state, beta_counts)
        return float(log_prob)

    def predictive_log_prob_new(
        self,
        observation: Union[
            Tuple[Optional[np.ndarray], Optional[Sequence[int]]],
            Tuple[Optional[np.ndarray], Optional[Sequence[int]], Optional[Tuple[int, int]]],
        ],
    ) -> float:
        temp_state = self.create_state()
        return self.predictive_log_prob(temp_state, observation)

    @property
    def has_beta_binomial(self) -> bool:
        return self.beta_binomial_prior is not None


    def _gaussian_log_predictive(
        self, state: HybridStateStatistics, cont: np.ndarray
    ) -> float:
        prior = self.gaussian_prior
        if prior is None:
            raise ValueError("Gaussian prior not configured")
        d = self.continuous_dim
        n = state.count
        if n < 0:
            raise ValueError("State count became negative; check sampler bookkeeping")
        kappa_n = prior.kappa0 + n
        nu_n = prior.nu0 + n
        mu_n = (prior.kappa0 * prior.mu0 + state.sum_cont) / kappa_n
        lambda_n = (
            prior.psi0
            + state.sum_outer
            + prior.kappa0 * np.outer(prior.mu0, prior.mu0)
            - kappa_n * np.outer(mu_n, mu_n)
        )
        df = nu_n - d + 1
        if df <= 0:
            raise ValueError("Degrees of freedom for predictive Student-t must be positive")
        scale = (kappa_n + 1) / (kappa_n * df) * lambda_n
        return _multivariate_student_t_logpdf(cont, mu_n, scale, df)

    def _categorical_log_predictive(
        self, state: HybridStateStatistics, cats: Sequence[int]
    ) -> float:
        if len(cats) != len(self.categorical_priors):
            raise ValueError("Categorical observation dimensionality mismatch")
        log_prob = 0.0
        for idx, prior in enumerate(self.categorical_priors):
            counts = state.cat_counts[idx]
            total = counts.sum()
            numerator = counts[cats[idx]] + prior[cats[idx]]
            denominator = total + prior.sum()
            log_prob += np.log(numerator / denominator)
        return log_prob


    def _beta_binomial_log_predictive(
        self, state: HybridStateStatistics, beta_counts: Tuple[int, int]
    ) -> float:
        prior = self.beta_binomial_prior
        if prior is None:
            raise ValueError("Beta-Binomial prior not configured")
        successes, trials = beta_counts
        if trials < 0 or successes < 0 or successes > trials:
            raise ValueError("Invalid Beta-Binomial observation counts")
        alpha_post = prior.alpha0 + state.successes
        beta_post = prior.beta0 + state.trials - state.successes
        log_coeff = (
            ssp.gammaln(trials + 1)
            - ssp.gammaln(successes + 1)
            - ssp.gammaln(trials - successes + 1)
        )
        return float(
            log_coeff
            + ssp.gammaln(alpha_post + successes)
            + ssp.gammaln(beta_post + trials - successes)
            - ssp.gammaln(alpha_post + beta_post + trials)
            + ssp.gammaln(alpha_post + beta_post)
            - ssp.gammaln(alpha_post)
            - ssp.gammaln(beta_post)
        )


def _multivariate_student_t_logpdf(
    x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, df: float
) -> float:
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d = x.shape[0]
    diff = x - mu
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        raise np.linalg.LinAlgError("Predictive scale matrix is not positive definite")
    quad = diff.T @ np.linalg.solve(sigma, diff)
    log_norm = (
        ssp.gammaln((df + d) / 2.0)
        - ssp.gammaln(df / 2.0)
        - 0.5 * (d * np.log(df * np.pi) + logdet)
    )
    return float(log_norm - 0.5 * (df + d) * np.log1p(quad / df))


__all__ = [
    "GaussianNIWPrior",
    "BetaBinomialPrior",
    "HybridEmissionModel",
    "HybridStateStatistics",
]