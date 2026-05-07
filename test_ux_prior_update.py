# -*- coding: utf-8 -*-
"""
Pytest suite for ux_prior_update.

Covers:
    A. Step 3 conditional mean (worked example + general formula)
    B. Output validity (length, sign, normalisation)
    C. rho behaviour (independence at 0, tracking at high values)
    D. Symmetric input → zero conditional mean
    E. Extreme posteriors (all mass in one bin)
    F. Truncation edge handling
    G. Integration with MBNpy (build_temporal_bn + inference)
"""

import numpy as np
import pytest
from scipy import stats

from ux_prior_update import (
    BIN_EDGES,
    BIN_MIDPOINTS,
    _compute_conditional_mean,
    update_prior_for_next_step,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _all_mass_in_bin(idx: int) -> np.ndarray:
    """Return a length-5 posterior with all probability in bin `idx`."""
    p = np.zeros(5)
    p[idx] = 1.0
    return p


def _base_distribution() -> np.ndarray:
    """Truncnorm CDF differences centred at mu=0 (the rho=0 / zero-mean reference)."""
    a = (BIN_EDGES[0] - 0.0) / 1.0
    b = (BIN_EDGES[-1] - 0.0) / 1.0
    dist = stats.truncnorm(a, b, loc=0.0, scale=1.0)
    return np.diff(dist.cdf(BIN_EDGES))


# ---------------------------------------------------------------------------
# A. Step 3: conditional mean propagation
# ---------------------------------------------------------------------------

class TestConditionalMean:

    def test_step3_worked_example(self):
        """A1: spec worked example — rho=0.2, mu_post=0.75 → mu_next=0.15."""
        mu_next = _compute_conditional_mean(mu_post=0.75, rho=0.2)
        assert np.isclose(mu_next, 0.15, atol=1e-9), (
            f"Expected 0.15, got {mu_next}"
        )

    @pytest.mark.parametrize("rho", [0.0, 0.2, 0.5, 0.9])
    def test_step3_general_formula(self, rho):
        """A2: zero-mean unit-var 2×2 case always reduces to mu_next = rho * mu_post."""
        mu_post = 0.75
        mu_next = _compute_conditional_mean(mu_post=mu_post, rho=rho)
        assert np.isclose(mu_next, rho * mu_post, atol=1e-9), (
            f"rho={rho}: expected {rho * mu_post}, got {mu_next}"
        )

    def test_step3_negative_mu_post(self):
        """A3: signed values propagate correctly — linearity of the formula."""
        mu_next = _compute_conditional_mean(mu_post=-0.4, rho=0.5)
        assert np.isclose(mu_next, -0.2, atol=1e-9)


# ---------------------------------------------------------------------------
# B. Output validity
# ---------------------------------------------------------------------------

class TestOutputValidity:

    @pytest.mark.parametrize("rho", [0.0, 0.2, 0.5, 0.9])
    def test_output_length(self, rho):
        """B1: output has exactly 5 elements."""
        out = update_prior_for_next_step(_all_mass_in_bin(2), rho=rho)
        assert len(out) == 5

    @pytest.mark.parametrize("rho", [0.0, 0.2, 0.5, 0.9])
    def test_output_nonnegative(self, rho):
        """B2: all output probabilities are non-negative."""
        out = update_prior_for_next_step(_all_mass_in_bin(2), rho=rho)
        assert np.all(out >= 0), f"Negative values found: {out}"

    @pytest.mark.parametrize("bin_idx,rho", [
        (0, 0.0), (1, 0.2), (2, 0.5), (3, 0.9), (4, 0.2),
    ])
    def test_output_sums_to_one(self, bin_idx, rho):
        """B3: probabilities sum to exactly 1.0 (from truncnorm CDF differences)."""
        out = update_prior_for_next_step(_all_mass_in_bin(bin_idx), rho=rho)
        assert np.isclose(np.sum(out), 1.0, atol=1e-12), (
            f"Sum={np.sum(out):.15f} for bin_idx={bin_idx}, rho={rho}"
        )


# ---------------------------------------------------------------------------
# C. rho behaviour
# ---------------------------------------------------------------------------

class TestRhoBehaviour:

    def test_rho_zero_gives_base_distribution(self):
        """C1: rho=0 → mu_next=0 → output equals the zero-centred baseline,
        regardless of the posterior input."""
        for bin_idx in range(5):
            out = update_prior_for_next_step(_all_mass_in_bin(bin_idx), rho=0.0)
            np.testing.assert_allclose(
                out, _base_distribution(), atol=1e-12,
                err_msg=f"rho=0 output differs from baseline for bin_idx={bin_idx}",
            )

    @pytest.mark.parametrize("rho", [0.7, 0.9, 0.99])
    def test_rho_high_shifts_prior_toward_posterior(self, rho):
        """C2: high rho → prior shifts toward the posterior's direction.
        All mass in highest bin (mu_post=0.375 > 0) → upper bins gain relative
        to the zero-centred baseline."""
        out = update_prior_for_next_step(_all_mass_in_bin(4), rho=rho)
        base = _base_distribution()
        assert np.sum(out[3:]) > np.sum(base[3:]), (
            f"rho={rho}: expected upper-bin mass to exceed baseline"
        )


# ---------------------------------------------------------------------------
# D. Symmetric input → zero conditional mean
# ---------------------------------------------------------------------------

class TestSymmetricInput:

    def test_zero_mean_posterior_gives_base_distribution(self):
        """D1: posterior with E[U]=0 → mu_next=0 for any rho → output equals
        the zero-centred baseline.

        p = [0, 0, 0.5, 0.5, 0] has mean = 0.5*(-0.125) + 0.5*(0.125) = 0.
        """
        p = np.array([0.0, 0.0, 0.5, 0.5, 0.0])
        mu_post = float(np.dot(p, BIN_MIDPOINTS))
        assert np.isclose(mu_post, 0.0, atol=1e-12), (
            f"Test posterior does not have zero mean: {mu_post}"
        )
        base = _base_distribution()
        for rho in [0.0, 0.2, 0.9]:
            out = update_prior_for_next_step(p, rho=rho)
            np.testing.assert_allclose(
                out, base, atol=1e-12,
                err_msg=f"Zero-mean posterior gave non-base output for rho={rho}",
            )


# ---------------------------------------------------------------------------
# E. Extreme posteriors (all mass in one bin)
# ---------------------------------------------------------------------------

class TestExtremePosteriors:

    def test_mu_next_all_mass_lowest_bin(self):
        """E1a: all mass in bin 0, mu_post=-0.625, rho=0.2 → mu_next=-0.125."""
        mu_next = _compute_conditional_mean(mu_post=BIN_MIDPOINTS[0], rho=0.2)
        assert np.isclose(mu_next, 0.2 * BIN_MIDPOINTS[0], atol=1e-9)

    def test_mu_next_all_mass_highest_bin(self):
        """E2a: all mass in bin 4, mu_post=0.375, rho=0.2 → mu_next=0.075."""
        mu_next = _compute_conditional_mean(mu_post=BIN_MIDPOINTS[4], rho=0.2)
        assert np.isclose(mu_next, 0.2 * BIN_MIDPOINTS[4], atol=1e-9)

    def test_all_mass_lowest_bin_shifts_distribution_down(self):
        """E1b: extreme low posterior → output has more lower-bin mass than baseline."""
        out = update_prior_for_next_step(_all_mass_in_bin(0), rho=0.2)
        base = _base_distribution()
        assert np.sum(out[:2]) > np.sum(base[:2])

    def test_all_mass_highest_bin_shifts_distribution_up(self):
        """E2b: extreme high posterior → output has more upper-bin mass than baseline."""
        out = update_prior_for_next_step(_all_mass_in_bin(4), rho=0.2)
        base = _base_distribution()
        assert np.sum(out[3:]) > np.sum(base[3:])


# ---------------------------------------------------------------------------
# F. Truncation edge handling
# ---------------------------------------------------------------------------

class TestTruncationEdges:

    @pytest.mark.parametrize("rho", [0.0, 0.2, 0.5, 0.9])
    def test_no_probability_leaks_outside_truncation(self, rho):
        """F1: sum of output probabilities equals 1.0 for all bins and rho values
        — confirms no probability mass outside the truncation range."""
        for bin_idx in range(5):
            out = update_prior_for_next_step(_all_mass_in_bin(bin_idx), rho=rho)
            assert np.isclose(np.sum(out), 1.0, atol=1e-12), (
                f"Leak detected: sum={np.sum(out):.15f}, bin_idx={bin_idx}, rho={rho}"
            )

    def test_bin_edge_count_is_n_bins_plus_one(self):
        """F2a: BIN_EDGES has exactly one more element than BIN_MIDPOINTS."""
        assert len(BIN_EDGES) == len(BIN_MIDPOINTS) + 1

    def test_bin_edges_match_default_truncation_range(self):
        """F2b: outer BIN_EDGES equal the default trunc_range (-0.75, 0.5)."""
        assert BIN_EDGES[0] == -0.75
        assert BIN_EDGES[-1] == 0.5

    def test_bin_edges_are_strictly_increasing(self):
        """F2c: bin edges are strictly monotone — no zero-width or reversed bins."""
        assert np.all(np.diff(BIN_EDGES) > 0)


# ---------------------------------------------------------------------------
# G. Integration with MBNpy
# ---------------------------------------------------------------------------

class TestMBNpyIntegration:

    def test_output_usable_as_mbnpy_ux_prior(self):
        """G1: output is accepted by build_temporal_bn as p_ux without error,
        and is stored verbatim in the Ux CPM."""
        from cumulative_damage_bn_v2 import build_temporal_bn

        posterior = np.array([0.0, 0.006, 0.493, 0.493, 0.006])
        new_p_ux = update_prior_for_next_step(posterior, rho=0.2)

        varis, cpms = build_temporal_bn(n_components=1, p_ux=new_p_ux)
        # MBNpy stores .p as a (n, 1) column vector; flatten before comparing.
        np.testing.assert_allclose(cpms['Ux'].p.flatten(), new_p_ux, atol=1e-12)

    def test_full_cycle_posterior_to_prior_to_inference(self):
        """G2: end-to-end cycle — VE Ux posterior → update → second-step inference.

        Builds a 1-component BN, queries the Ux marginal via variable elimination,
        normalises, propagates to U_{x+1} prior, builds a second-step BN, and
        verifies the second-step inference completes with a valid marginal.
        """
        from mbnpy import inference
        from cumulative_damage_bn_v2 import build_temporal_bn, build_elimination_order

        # Step 1: build first-step BN and query the Ux marginal
        varis1, cpms1 = build_temporal_bn(n_components=1)
        ux_query_order = [
            varis1['Wx1'], varis1['Lx1'], varis1['Tx1'],
            varis1['Px1'], varis1['Vx1'], varis1['Zx1'],
            varis1['Rx1'], varis1['Clx1'], varis1['Cx0'], varis1['Cx1'],
        ]
        ux_result = inference.variable_elim(
            cpms=cpms1, var_elim=ux_query_order, prod=True
        )
        p_ux_post = np.array(ux_result.p, dtype=float)
        p_ux_post /= p_ux_post.sum()   # normalise factor → probability vector

        # Step 2: propagate to U_{x+1} prior
        p_ux_next = update_prior_for_next_step(p_ux_post, rho=0.2)

        # Step 3: build second-step BN with updated prior and run inference
        varis2, cpms2 = build_temporal_bn(n_components=1, p_ux=p_ux_next)
        order2 = build_elimination_order(varis2, 1)
        result2 = inference.variable_elim(cpms=cpms2, var_elim=order2, prod=True)

        assert result2 is not None
        assert hasattr(result2, 'p')
        assert abs(float(np.sum(result2.p)) - 1.0) < 0.01
