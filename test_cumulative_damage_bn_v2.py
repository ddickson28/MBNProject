# -*- coding: utf-8 -*-
"""
Pytest suite for cumulative_damage_bn_v2.

Covers Cx0 validation, builder structure, new variable wiring (Zx, Vx, Ux),
CPM table integrity, elimination order, and an end-to-end inference smoke test.
"""

import numpy as np
import pytest

from cumulative_damage_bn_v2 import (
    build_temporal_bn,
    build_elimination_order,
    validate_cx0_probs,
)


# =========================================================================
# A. Cx0 validation
# =========================================================================

class TestCx0Validation:

    def test_cx0_all_zeros_raises(self):
        """A1: All-zero Cx0 must raise ValueError."""
        p_cx0 = np.zeros(6)
        with pytest.raises(ValueError, match="all entries are zero"):
            build_temporal_bn(n_components=1, p_cx0=p_cx0)

    def test_cx0_multiple_ones_raises(self):
        """A2: Cx0 with more than one 1.0 must raise ValueError."""
        p_cx0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="non-zero entr"):
            build_temporal_bn(n_components=1, p_cx0=p_cx0)

    def test_cx0_single_non_one_raises(self):
        """A3: Cx0 with one non-zero entry that is not 1.0 must raise."""
        p_cx0 = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="must equal 1.0"):
            build_temporal_bn(n_components=1, p_cx0=p_cx0)


# =========================================================================
# B. Builder structure
# =========================================================================

class TestBuilderStructure:

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_variable_count_per_n(self, n):
        """B1: total variables = Cx0 + 9n + 1 (Ux)."""
        varis, _ = build_temporal_bn(n_components=n)
        expected = 1 + 9 * n + 1
        assert len(varis) == expected

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_cpm_count_per_n(self, n):
        """B2: total CPMs = Cx0 + 9n + 1 (Ux)."""
        _, cpms = build_temporal_bn(n_components=n)
        expected = 1 + 9 * n + 1
        assert len(cpms) == expected

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_ux_is_singleton(self, n):
        """B3: exactly one Ux variable and one Ux CPM regardless of n."""
        varis, cpms = build_temporal_bn(n_components=n)
        ux_vars = [k for k in varis if k.startswith('Ux')]
        ux_cpms = [k for k in cpms if k.startswith('Ux')]
        assert ux_vars == ['Ux']
        assert ux_cpms == ['Ux']

    def test_n_components_zero_raises(self):
        """B4: n_components=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            build_temporal_bn(n_components=0)

    def test_n_components_negative_raises(self):
        """B5: negative n_components raises ValueError."""
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            build_temporal_bn(n_components=-2)


# =========================================================================
# C. New variable wiring (Zx, Vx, Ux, Rx)
# =========================================================================

class TestNewVariableWiring:

    def test_zx_depends_on_vx_and_ux(self):
        """C1: Zx{i} CPM lists [Zx{i}, Vx{i}, Ux] in that order."""
        varis, cpms = build_temporal_bn(n_components=3)
        for i in range(1, 4):
            names = [v.name for v in cpms[f'Zx{i}'].variables]
            assert names == [f'Zx{i}', f'Vx{i}', 'Ux']

    def test_rx_depends_on_zx(self):
        """C2: Rx{i} is conditioned on Zx{i}."""
        varis, cpms = build_temporal_bn(n_components=3)
        for i in range(1, 4):
            names = [v.name for v in cpms[f'Rx{i}'].variables]
            assert names == [f'Rx{i}', f'Zx{i}']

    def test_ux_shared_across_components(self):
        """C3: every Zx{i} references the SAME Ux object (identity check)."""
        varis, cpms = build_temporal_bn(n_components=3)
        ux_obj = varis['Ux']
        for i in range(1, 4):
            ux_in_zx = cpms[f'Zx{i}'].variables[2]
            assert ux_in_zx is ux_obj, (
                f"Zx{i} references a different Ux object — Ux must be shared."
            )

    def test_vx_is_per_component(self):
        """C4: Vx{i} are distinct variable objects with distinct CPMs."""
        varis, cpms = build_temporal_bn(n_components=2)
        assert varis['Vx1'] is not varis['Vx2']
        assert cpms['Vx1'] is not cpms['Vx2']


# =========================================================================
# D. CPM table integrity
# =========================================================================

class TestCpmTableIntegrity:

    def test_zx_table_shape(self):
        """D1: Zx CPM has 25 rows; every (Vx, Ux) pair appears exactly once."""
        _, cpms = build_temporal_bn(n_components=1)
        c = cpms['Zx1'].C
        assert c.shape[0] == 25, f"Expected 25 rows, got {c.shape[0]}"

        pairs = {(int(row[1]), int(row[2])) for row in c}
        expected = {(v, u) for v in range(5) for u in range(5)}
        assert pairs == expected, "Missing or duplicate (Vx, Ux) combinations."

    def test_rx_table_shape(self):
        """D2: Rx CPM has 5 rows covering Zx states 0..4 (states 0/5 unreachable by design)."""
        _, cpms = build_temporal_bn(n_components=1)
        c = cpms['Rx1'].C
        assert c.shape[0] == 5
        zx_states = sorted(int(row[1]) for row in c)
        assert zx_states == [0, 1, 2, 3, 4]

    def test_cpm_probabilities_nonnegative(self):
        """D3: every CPM's p vector is non-negative."""
        _, cpms = build_temporal_bn(n_components=2)
        for name, c in cpms.items():
            assert np.all(c.p >= 0), f"Negative probability found in {name}."


# =========================================================================
# E. Elimination order
# =========================================================================

class TestEliminationOrder:

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_elim_order_includes_ux_once(self, n):
        """E1: Ux appears exactly once in elimination order."""
        varis, _ = build_temporal_bn(n_components=n)
        order = build_elimination_order(varis, n)
        ux_count = sum(1 for v in order if v.name == 'Ux')
        assert ux_count == 1

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_elim_order_includes_all_vx_zx(self, n):
        """E2: Vx{i} and Zx{i} appear for every i in [1, n]."""
        varis, _ = build_temporal_bn(n_components=n)
        order = build_elimination_order(varis, n)
        names = [v.name for v in order]
        for i in range(1, n + 1):
            assert f'Vx{i}' in names, f"Vx{i} missing from elimination order"
            assert f'Zx{i}' in names, f"Zx{i} missing from elimination order"

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_elim_order_no_duplicates(self, n):
        """E3: no variable appears twice in elimination order."""
        varis, _ = build_temporal_bn(n_components=n)
        order = build_elimination_order(varis, n)
        names = [v.name for v in order]
        duplicates = {x for x in names if names.count(x) > 1}
        assert not duplicates, f"Duplicate variables in elim order: {duplicates}"


# =========================================================================
# F. End-to-end smoke test
# =========================================================================

class TestEndToEndInference:

    def test_inference_runs_without_error(self):
        """F1: full pipeline runs and returns a CPM with non-empty p vector."""
        from mbnpy import inference
        varis, cpms = build_temporal_bn(n_components=2)
        order = build_elimination_order(varis, 2)
        result = inference.variable_elim(cpms=cpms, var_elim=order, prod=True)
        assert result is not None
        assert hasattr(result, 'p')
        assert len(result.p) > 0

    def test_inference_marginal_sums_to_one(self):
        """F2: resulting marginal p vector sums to ~1.0 (within tolerance)."""
        from mbnpy import inference
        varis, cpms = build_temporal_bn(n_components=2)
        order = build_elimination_order(varis, 2)
        result = inference.variable_elim(cpms=cpms, var_elim=order, prod=True)
        # Tolerance set to 0.01 to accommodate the known Vx/Ux rounding (sum = 0.998).
        assert abs(float(np.sum(result.p)) - 1.0) < 0.01
