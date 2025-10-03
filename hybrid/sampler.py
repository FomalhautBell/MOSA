"""DS-HDP-HMM sampler supporting hybrid continuous/categorical emissions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import beta, binomial, dirichlet, gamma, multinomial

from .emissions import HybridEmissionModel, HybridStateStatistics


def transform_var_poly(v0: float, v1: float, p: float) -> Tuple[float, float]:
    """Map auxiliary variables to Beta prior parameters."""

    if p == float("inf") or p == "inf":
        rho0 = -v0 * np.log(v1)
    else:
        rho0 = v0 / np.power(v1, float(p))
    rho1 = (1.0 - v0) * rho0 / v0
    return float(rho0), float(rho1)


@dataclass
class HybridObservations:
    """Container for hybrid observation sequences."""

    continuous: Optional[np.ndarray]
    categorical: Optional[Sequence[np.ndarray]]

    def __post_init__(self) -> None:
        if self.continuous is None and not self.categorical:
            raise ValueError("At least one observation modality must be provided")
        lengths: List[int] = []
        if self.continuous is not None:
            self.continuous = np.asarray(self.continuous, dtype=float)
            if self.continuous.ndim == 1:
                self.continuous = self.continuous[:, None]
            lengths.append(int(self.continuous.shape[0]))
        if self.categorical:
            processed: List[np.ndarray] = []
            for arr in self.categorical:
                cat_arr = np.asarray(arr, dtype=int)
                lengths.append(int(len(cat_arr)))
                processed.append(cat_arr)
            self.categorical = processed
        if len(set(lengths)) != 1:
            raise ValueError("Continuous and categorical observations must share the same length")
        self.length = lengths[0]

    def get(self, index: int) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        cont = None if self.continuous is None else self.continuous[index]
        cats = None
        if self.categorical:
            cats = [int(arr[index]) for arr in self.categorical]
        return cont, cats


class DSHDPHMMHybridSampler:
    """Direct-assignment Gibbs sampler for hybrid-emission DS-HDP-HMM."""

    def __init__(
        self,
        observations: HybridObservations,
        emission_model: HybridEmissionModel,
        alpha0: float,
        gamma0: float,
        rho0: float,
        rho1: float,
    ) -> None:
        self.obs = observations
        self.model = emission_model
        self.alpha0 = float(alpha0)
        self.gamma0 = float(gamma0)
        self.rho0 = float(rho0)
        self.rho1 = float(rho1)
        self.reset_state()

    def reset_state(self) -> None:
        self.K = 0
        self.zt = np.zeros(self.obs.length, dtype=int)
        self.wt = np.zeros(self.obs.length, dtype=int)
        self.beta_vec = np.array([], dtype=float)
        self.beta_new = 1.0
        self.kappa_vec = np.array([], dtype=float)
        self.kappa_new = 0.5
        self.n_mat = np.zeros((0, 0), dtype=float)
        self.states: List[HybridStateStatistics] = []

    def initialize(self) -> None:
        self.reset_state()
        T = self.obs.length
        self.K = 1
        self.zt = np.zeros(T, dtype=int)
        beta_draw = dirichlet(np.array([1.0, self.gamma0]))
        self.beta_vec = np.array([beta_draw[0]])
        self.beta_new = float(beta_draw[1])
        kappa_init = beta(self.rho0, self.rho1, size=1)[0]
        self.kappa_vec = np.array([kappa_init])
        self.kappa_new = beta(self.rho0, self.rho1, size=1)[0]
        self.kappa_vec = np.clip(self.kappa_vec, 0.0, 0.8)
        self.wt = binomial(1, self.kappa_vec[0], size=T)
        self.wt[0] = 0
        self.n_mat = np.array([[0.0]])
        self.states = [self.model.create_state()]
        self.states[0].add_observation(self.obs.get(0))
        (
            self.zt,
            self.wt,
            self.n_mat,
            self.states,
            self.beta_vec,
            self.beta_new,
            self.kappa_vec,
            self.kappa_new,
            self.K,
        ) = sample_one_step_ahead(
            self.obs,
            self.model,
            self.zt,
            self.wt,
            self.n_mat,
            self.states,
            self.beta_vec,
            self.beta_new,
            self.kappa_vec,
            self.kappa_new,
            self.alpha0,
            self.gamma0,
            self.rho0,
            self.rho1,
            self.K,
        )

    def step(
        self,
        resample_hyperparams: bool = False,
        alpha_prior: Optional[Tuple[float, float]] = None,
        gamma_prior: Optional[Tuple[float, float]] = None,
        rho_grid: Optional[Dict[str, float]] = None,
    ) -> None:
        (
            self.zt,
            self.wt,
            self.n_mat,
            self.states,
            self.beta_vec,
            self.beta_new,
            self.kappa_vec,
            self.kappa_new,
            self.K,
        ) = sample_zw(
            self.obs,
            self.model,
            self.zt,
            self.wt,
            self.n_mat,
            self.states,
            self.beta_vec,
            self.beta_new,
            self.kappa_vec,
            self.kappa_new,
            self.alpha0,
            self.gamma0,
            self.rho0,
            self.rho1,
            self.K,
        )
        (
            self.zt,
            self.n_mat,
            self.states,
            self.beta_vec,
            self.K,
        ) = decre_K(self.zt, self.n_mat, self.states, self.beta_vec)
        self.kappa_vec, self.kappa_new, num_1_vec, num_0_vec = sample_kappa(
            self.zt, self.wt, self.rho0, self.rho1, self.K
        )
        m_mat = sample_m(self.n_mat, self.beta_vec, self.alpha0, self.K)
        self.beta_vec, self.beta_new = sample_beta(m_mat, self.gamma0)
        if resample_hyperparams:
            if alpha_prior is None or gamma_prior is None or rho_grid is None:
                raise ValueError("Hyper-parameter priors must be provided for resampling")
            self.alpha0 = sample_alpha(
                m_mat, self.n_mat, self.alpha0, alpha_prior[0], alpha_prior[1]
            )
            self.gamma0 = sample_gamma(self.K, m_mat, self.gamma0, gamma_prior[0], gamma_prior[1])
            self.rho0, self.rho1, _ = sample_rho(
                rho_grid["v0_range"],
                rho_grid["v1_range"],
                int(rho_grid["v0_num_grid"]),
                int(rho_grid["v1_num_grid"]),
                self.K,
                num_1_vec,
                num_0_vec,
                rho_grid["p"],
            )

    def run(
        self,
        num_iterations: int,
        burn_in: int = 0,
        resample_hyperparams: bool = False,
        alpha_prior: Optional[Tuple[float, float]] = None,
        gamma_prior: Optional[Tuple[float, float]] = None,
        rho_grid: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[np.ndarray]]:
        if self.K == 0:
            self.initialize()
        history: Dict[str, List[np.ndarray]] = {"zt": [], "wt": []}
        for it in range(num_iterations):
            self.step(
                resample_hyperparams=resample_hyperparams,
                alpha_prior=alpha_prior,
                gamma_prior=gamma_prior,
                rho_grid=rho_grid,
            )
            if it >= burn_in:
                history["zt"].append(self.zt.copy())
                history["wt"].append(self.wt.copy())
        return history


def sample_one_step_ahead(
    obs: HybridObservations,
    model: HybridEmissionModel,
    zt: np.ndarray,
    wt: np.ndarray,
    n_mat: np.ndarray,
    states: List[HybridStateStatistics],
    beta_vec: np.ndarray,
    beta_new: float,
    kappa_vec: np.ndarray,
    kappa_new: float,
    alpha0: float,
    gamma0: float,
    rho0: float,
    rho1: float,
    K: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[HybridStateStatistics],
    np.ndarray,
    float,
    np.ndarray,
    float,
    int,
]:
    T = len(zt)
    for t in range(1, T):
        j = zt[t - 1]
        observation = obs.get(t)
        zt_dist = (alpha0 * beta_vec + n_mat[j]) / (alpha0 + n_mat[j].sum())
        yt_dist = np.array(
            [np.exp(state.predictive_log_prob(observation)) for state in states]
        )
        knew_dist = alpha0 * beta_new / (alpha0 + n_mat[j].sum())
        yt_knew = np.exp(model.predictive_log_prob_new(observation))
        post_cases = np.hstack(
            (
                kappa_vec[j] * yt_dist[j],
                (1 - kappa_vec[j]) * zt_dist * yt_dist,
                (1 - kappa_vec[j]) * knew_dist * yt_knew,
            )
        )
        post_cases = post_cases / post_cases.sum()
        sample_rlt = np.where(multinomial(1, post_cases))[0][0]
        if sample_rlt < 1:
            zt[t], wt[t] = j, 1
        else:
            zt[t], wt[t] = sample_rlt - 1, 0
        if zt[t] == K:
            b = beta(1.0, gamma0, size=1)[0]
            beta_vec = np.hstack((beta_vec, b * beta_new))
            kappa_vec = np.hstack((kappa_vec, kappa_new))
            beta_new = (1 - b) * beta_new
            kappa_new = beta(rho0, rho1, size=1)[0]
            kappa_vec[-1] = np.clip(kappa_vec[-1], 0.0, 0.8)
            states.append(model.create_state())
            n_mat = np.hstack((n_mat, np.zeros((K, 1))))
            n_mat = np.vstack((n_mat, np.zeros((1, K + 1))))
            K += 1
        if wt[t] == 0:
            n_mat[j, zt[t]] += 1
        states[zt[t]].add_observation(observation)
    return zt, wt, n_mat, states, beta_vec, beta_new, kappa_vec, kappa_new, K


def sample_last(
    obs: HybridObservations,
    model: HybridEmissionModel,
    zt: np.ndarray,
    wt: np.ndarray,
    n_mat: np.ndarray,
    states: List[HybridStateStatistics],
    beta_vec: np.ndarray,
    beta_new: float,
    kappa_vec: np.ndarray,
    kappa_new: float,
    alpha0: float,
    gamma0: float,
    rho0: float,
    rho1: float,
    K: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[HybridStateStatistics],
    np.ndarray,
    float,
    np.ndarray,
    float,
    int,
]:
    t = len(zt) - 1
    j = zt[t - 1]
    observation = obs.get(t)
    if wt[t] == 0:
        n_mat[j, zt[t]] -= 1
    states[zt[t]].remove_observation(observation)
    zt_dist = (alpha0 * beta_vec + n_mat[j]) / (alpha0 + n_mat[j].sum())
    yt_dist = np.array(
        [np.exp(state.predictive_log_prob(observation)) for state in states]
    )
    knew_dist = alpha0 * beta_new / (alpha0 + n_mat[j].sum())
    yt_knew = np.exp(model.predictive_log_prob_new(observation))
    post_cases = np.hstack(
        (
            kappa_vec[j] * yt_dist[j],
            (1 - kappa_vec[j]) * zt_dist * yt_dist,
            (1 - kappa_vec[j]) * knew_dist * yt_knew,
        )
    )
    post_cases = post_cases / post_cases.sum()
    sample_rlt = np.where(multinomial(1, post_cases))[0][0]
    if sample_rlt < 1:
        zt[t], wt[t] = j, 1
    else:
        zt[t], wt[t] = sample_rlt - 1, 0
    if zt[t] == K:
        b = beta(1.0, gamma0, size=1)[0]
        beta_vec = np.hstack((beta_vec, b * beta_new))
        kappa_vec = np.hstack((kappa_vec, kappa_new))
        beta_new = (1 - b) * beta_new
        kappa_new = beta(rho0, rho1, size=1)[0]
        kappa_vec[-1] = np.clip(kappa_vec[-1], 0.0, 0.8)
        states.append(model.create_state())
        n_mat = np.hstack((n_mat, np.zeros((K, 1))))
        n_mat = np.vstack((n_mat, np.zeros((1, K + 1))))
        K += 1
    if wt[t] == 0:
        n_mat[j, zt[t]] += 1
    states[zt[t]].add_observation(observation)
    return zt, wt, n_mat, states, beta_vec, beta_new, kappa_vec, kappa_new, K


def sample_zw(
    obs: HybridObservations,
    model: HybridEmissionModel,
    zt: np.ndarray,
    wt: np.ndarray,
    n_mat: np.ndarray,
    states: List[HybridStateStatistics],
    beta_vec: np.ndarray,
    beta_new: float,
    kappa_vec: np.ndarray,
    kappa_new: float,
    alpha0: float,
    gamma0: float,
    rho0: float,
    rho1: float,
    K: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[HybridStateStatistics],
    np.ndarray,
    float,
    np.ndarray,
    float,
    int,
]:
    T = len(zt)
    tmp_vec = np.arange(K)
    for t in range(1, T - 1):
        j = zt[t - 1]
        l = zt[t + 1]
        observation = obs.get(t)
        if wt[t] == 0:
            n_mat[j, zt[t]] -= 1
        if wt[t + 1] == 0:
            n_mat[zt[t], l] -= 1
        states[zt[t]].remove_observation(observation)
        zt_dist = (alpha0 * beta_vec + n_mat[j]) / (alpha0 + n_mat[j].sum())
        ztplus1_dist = (
            alpha0 * beta_vec[l]
            + n_mat[:, l]
            + (j == l) * (j == tmp_vec)
        ) / (alpha0 + n_mat.sum(axis=1) + (j == tmp_vec))
        yt_dist = np.array(
            [np.exp(state.predictive_log_prob(observation)) for state in states]
        )
        knew_dist = (alpha0**2) * beta_vec[l] * beta_new / (
            alpha0 * (alpha0 + n_mat[j].sum())
        )
        yt_knew = np.exp(model.predictive_log_prob_new(observation))
        post_cases = np.array(
            (
                (kappa_vec[j] ** 2) * yt_dist[j] * (j == l),
                (1 - kappa_vec[j]) * kappa_vec[l] * zt_dist[l] * yt_dist[l],
                kappa_vec[j] * (1 - kappa_vec[j]) * ztplus1_dist[j] * yt_dist[j],
            )
        )
        post_cases = np.hstack(
            (
                post_cases,
                (1 - kappa_vec[j])
                * (1 - kappa_vec)
                * zt_dist
                * ztplus1_dist
                * yt_dist,
                (1 - kappa_vec[j]) * (1 - kappa_new) * knew_dist * yt_knew,
            )
        )
        post_cases = post_cases / post_cases.sum()
        sample_rlt = np.where(multinomial(1, post_cases))[0][0]
        if sample_rlt < 3:
            choices = [[j, 1, 1], [l, 0, 1], [j, 1, 0]]
            zt[t], wt[t], wt[t + 1] = choices[sample_rlt]
        else:
            zt[t], wt[t], wt[t + 1] = sample_rlt - 3, 0, 0
        if zt[t] == K:
            b = beta(1.0, gamma0, size=1)[0]
            beta_vec = np.hstack((beta_vec, b * beta_new))
            kappa_vec = np.hstack((kappa_vec, kappa_new))
            beta_new = (1 - b) * beta_new
            kappa_new = beta(rho0, rho1, size=1)[0]
            kappa_vec[-1] = np.clip(kappa_vec[-1], 0.0, 0.8)
            states.append(model.create_state())
            n_mat = np.hstack((n_mat, np.zeros((K, 1))))
            n_mat = np.vstack((n_mat, np.zeros((1, K + 1))))
            tmp_vec = np.arange(K + 1)
            K += 1
        if wt[t] == 0:
            n_mat[j, zt[t]] += 1
        if wt[t + 1] == 0:
            n_mat[zt[t], l] += 1
        states[zt[t]].add_observation(observation)
    return sample_last(
        obs,
        model,
        zt,
        wt,
        n_mat,
        states,
        beta_vec,
        beta_new,
        kappa_vec,
        kappa_new,
        alpha0,
        gamma0,
        rho0,
        rho1,
        K,
    )


def decre_K(
    zt: np.ndarray,
    n_mat: np.ndarray,
    states: List[HybridStateStatistics],
    beta_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[HybridStateStatistics], np.ndarray, int]:
    rem_ind = np.unique(zt)
    mapping = {old: new for new, old in enumerate(sorted(rem_ind))}
    zt = np.array([mapping[x] for x in zt])
    n_mat = n_mat[rem_ind][:, rem_ind]
    beta_vec = beta_vec[rem_ind]
    states = [states[idx] for idx in rem_ind]
    return zt, n_mat, states, beta_vec, len(rem_ind)


def sample_kappa(
    zt: np.ndarray,
    wt: np.ndarray,
    rho0: float,
    rho1: float,
    K: int,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    kappa_vec = np.zeros(K)
    num_1_vec = np.zeros(K)
    num_0_vec = np.zeros(K)
    for j in range(K):
        ind_lists = np.where(zt[:-1] == j)[0] + 1
        num_1 = wt[ind_lists].sum()
        num_0 = len(ind_lists) - num_1
        num_1_vec[j] = num_1
        num_0_vec[j] = num_0
        kappa_vec[j] = beta(rho0 + num_1, rho1 + num_0, size=1)[0]
    kappa_new = beta(rho0, rho1, size=1)[0]
    return kappa_vec, kappa_new, num_1_vec, num_0_vec


def sample_m(n_mat: np.ndarray, beta_vec: np.ndarray, alpha0: float, K: int) -> np.ndarray:
    m_mat = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            if n_mat[j, k] == 0:
                continue
            probs = alpha0 * beta_vec[k] / (np.arange(n_mat[j, k]) + alpha0 * beta_vec[k])
            draws = binomial(1, probs)
            m_mat[j, k] = draws.sum()
    m_mat[0, 0] += 1
    return m_mat


def sample_beta(m_mat: np.ndarray, gamma0: float) -> Tuple[np.ndarray, float]:
    beta_full = dirichlet(np.hstack((m_mat.sum(axis=0), gamma0)))
    return beta_full[:-1], float(beta_full[-1])


def sample_alpha(
    m_mat: np.ndarray,
    n_mat: np.ndarray,
    alpha0: float,
    alpha0_a_pri: float,
    alpha0_b_pri: float,
) -> float:
    r_vec = []
    tmp = n_mat.sum(axis=1)
    for val in tmp:
        if val > 0:
            r_vec.append(beta(alpha0 + 1, val))
    r_vec = np.array(r_vec)
    s_vec = binomial(1, n_mat.sum(axis=1) / (n_mat.sum(axis=1) + alpha0))
    alpha0 = gamma(
        alpha0_a_pri + m_mat.sum() - 1 - sum(s_vec),
        1 / (alpha0_b_pri - sum(np.log(r_vec + 1e-6))),
    )
    return float(alpha0)


def sample_gamma(
    K: int,
    m_mat: np.ndarray,
    gamma0: float,
    gamma0_a_pri: float,
    gamma0_b_pri: float,
) -> float:
    eta = beta(gamma0 + 1, m_mat.sum())
    pi_m = (gamma0_a_pri + K - 1) / (
        gamma0_a_pri + K - 1 + m_mat.sum() * (gamma0_b_pri - np.log(eta + 1e-6))
    )
    indicator = binomial(1, pi_m)
    if indicator:
        gamma0 = gamma(
            gamma0_a_pri + K,
            1 / (gamma0_b_pri - np.log(eta + 1e-6)),
        )
    else:
        gamma0 = gamma(
            gamma0_a_pri + K - 1,
            1 / (gamma0_b_pri - np.log(eta + 1e-6)),
        )
    return float(gamma0)


def compute_rho_posterior(
    rho0: float,
    rho1: float,
    K: int,
    num_1_vec: np.ndarray,
    num_0_vec: np.ndarray,
) -> float:
    import scipy.special as ssp

    log_posterior = (
        K * (ssp.gammaln(rho0 + rho1) - ssp.gammaln(rho0) - ssp.gammaln(rho1))
        + np.sum(ssp.gammaln(rho0 + num_1_vec))
        + np.sum(ssp.gammaln(rho1 + num_0_vec))
        - np.sum(ssp.gammaln(rho0 + rho1 + num_1_vec + num_0_vec))
    )
    return float(np.real(log_posterior))


def sample_rho(
    v0_range: Tuple[float, float],
    v1_range: Tuple[float, float],
    v0_num_grid: int,
    v1_num_grid: int,
    K: int,
    num_1_vec: np.ndarray,
    num_0_vec: np.ndarray,
    p: float,
) -> Tuple[float, float, np.ndarray]:
    v0_grid = np.linspace(v0_range[0], v0_range[1], v0_num_grid)
    v1_grid = np.linspace(v1_range[0], v1_range[1], v1_num_grid)
    posterior_grid = np.zeros((v0_num_grid, v1_num_grid))
    for ii, v0 in enumerate(v0_grid):
        for jj, v1 in enumerate(v1_grid):
            rho0, rho1 = transform_var_poly(v0, v1, p)
            posterior_grid[ii, jj] = compute_rho_posterior(rho0, rho1, K, num_1_vec, num_0_vec)
    posterior_grid = np.exp(posterior_grid - posterior_grid.max())
    posterior_grid /= posterior_grid.sum()
    sample_idx = np.where(multinomial(1, posterior_grid.reshape(-1)))[0][0]
    v0 = v0_grid[int(sample_idx // v1_num_grid)]
    v1 = v1_grid[int(sample_idx % v1_num_grid)]
    rho0, rho1 = transform_var_poly(v0, v1, p)
    return float(rho0), float(rho1), posterior_grid


__all__ = [
    "HybridObservations",
    "DSHDPHMMHybridSampler",
    "transform_var_poly",
]