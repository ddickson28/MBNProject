# -*- coding: utf-8 -*-
"""
Temporal Ux prior propagation.

Propagates the posterior over U_x (a discrete, truncated-normal-based
variable) into a prior probability vector for U_{x+1}, using Gaussian
conditioning (Bishop 2.81) and truncated-normal CDF differences.
"""

import numpy as np
from scipy import stats

# Default bin geometry for the 5-state noise variable (Ux / Vx).
# Edges:     lower bound of bin 0 → upper bound of bin 4.
# Midpoints: geometric centre of each bin, used for E[U|E] computation.
# The labelled states ['-0.5', '-0.25', '0.0', '0.25', '0.5'] are bin
# identifiers, NOT midpoints — do not use them for expectation calculations.
BIN_EDGES     = np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5])
BIN_MIDPOINTS = np.array([-0.625, -0.375, -0.125, 0.125, 0.375])


def _compute_conditional_mean(mu_post: float, rho: float) -> float:
    """Return E[U_{x+1} | U_x = mu_post] for a zero-mean joint Gaussian.

    Applies Bishop (2.81) to the 2×2 system
        Σ = [[1, ρ], [ρ, 1]],   Λ = Σ⁻¹
    with both prior means equal to zero. In this special case the formula
    reduces to ρ · mu_post, but is computed via the full precision matrix
    so it generalises cleanly if the covariance structure is extended.

    Args:
        mu_post: Posterior mean of U_x (Step 2 output).
        rho:     Temporal correlation coefficient; must satisfy |rho| < 1.

    Returns:
        Conditional prior mean for U_{x+1}.
    """
    sigma_mat = np.array([[1.0, rho], [rho, 1.0]])
    lam = np.linalg.inv(sigma_mat)          # precision matrix Λ
    # Bishop 2.81:  μ_{b|a} = μ_b − Λ_bb⁻¹ Λ_ba (x_a − μ_a)
    # a = index 0 (U_x), b = index 1 (U_{x+1}), μ_a = μ_b = 0
    return float(-(1.0 / lam[1, 1]) * lam[1, 0] * mu_post)


def update_prior_for_next_step(
    posterior_probs: np.ndarray,
    bin_midpoints: np.ndarray = BIN_MIDPOINTS,
    bin_edges: np.ndarray = BIN_EDGES,
    rho: float = 0.2,
    sigma: float = 1.0,
    trunc_range: tuple[float, float] = (-0.75, 0.5),
) -> np.ndarray:
    """Compute the prior probability vector for U_{x+1} from the posterior over U_x.

    Step 2 — posterior mean:
        mu_post = Σᵢ P(binᵢ | E) · midpointᵢ

    Step 3 — Gaussian conditioning (Bishop 2.81):
        mu_next = −Λ_bb⁻¹ · Λ_ba · mu_post   (= ρ · mu_post for zero-mean unit-var case)

    Step 4 — truncated normal CDF differences:
        P(binᵢ) = CDF(upper_edgeᵢ) − CDF(lower_edgeᵢ)
        where CDF is truncnorm(mean=mu_next, std=sigma, bounds=trunc_range).

    Args:
        posterior_probs: Length-5 posterior probability vector over U_x bins.
        bin_midpoints:   Geometric centres of each bin (length 5).
        bin_edges:       Bin boundaries including outer edges (length 6).
        rho:             Temporal correlation between consecutive U nodes.
        sigma:           Marginal standard deviation (default 1.0).
        trunc_range:     (lower, upper) truncation bounds.

    Returns:
        Length-5 probability vector for U_{x+1}, summing to 1.
    """
    # Step 2 — posterior mean from discrete distribution
    # Flatten to handle MBNpy's (n, 1) column-vector convention for .p arrays.
    posterior_probs = np.asarray(posterior_probs).flatten()
    mu_post = float(np.dot(posterior_probs, bin_midpoints))

    # Step 3 — propagate to conditional prior mean for next step
    mu_next = _compute_conditional_mean(mu_post, rho)

    # Step 4 — truncated normal CDF differences over bin edges
    a_std = (trunc_range[0] - mu_next) / sigma
    b_std = (trunc_range[1] - mu_next) / sigma
    dist = stats.truncnorm(a_std, b_std, loc=mu_next, scale=sigma)

    return np.diff(dist.cdf(bin_edges))
