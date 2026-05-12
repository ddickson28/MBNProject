# -*- coding: utf-8 -*-
"""
Pytest suite for FPSOBN (FPSO cumulative-damage Bayesian Network).

Covers:
    A. Variable creation  – counts, names, state spaces
    B. CPM matrices       – shapes, index bounds, normalisation
    C. Network structure  – parent-child relationships
    D. Elimination order  – completeness, no duplicates, query excluded
    E. Inference          – known result for 2×2 case
    F. Temporal chain     – Cx{loc}{t} references Cx{loc}{t-1}
    G. Ux per time-step   – distinct Ux{t}, Zx wired to correct Ux{t}
    H. Scalability        – 1×1 and 3×2 networks complete without error
"""

import numpy as np
import pytest
from mbnpy import variable, cpm, inference


# ---------------------------------------------------------------------------
# BN builder (mirrors FPSOBN.ipynb exactly)
# ---------------------------------------------------------------------------

STATES_DAMAGE = ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
STATES_BOOL   = ['False', 'True']
STATES_RESIST = ['0.4', '0.6', '0.8', '1.0', '1.2', '1.4']
STATES_AUX    = ['-0.5', '-0.25', '0.0', '0.25', '0.5']

C_Tx = np.array([
    [0,0,0],[1,1,0],[2,2,0],[3,3,0],[4,4,0],[5,5,0],
    [1,0,1],[2,1,1],[3,2,1],[4,3,1],[5,4,1],[5,5,1],
    [0,0,2],[3,1,2],[4,2,2],[5,3,2],[5,4,2],[5,5,2],
    [3,0,3],[4,1,3],[5,2,3],[5,3,3],[5,4,3],[5,5,3],
    [4,0,4],[5,1,4],[5,2,4],[5,3,4],[5,4,4],[5,5,4],
    [5,0,5],[2,1,5],[5,2,5],[5,3,5],[5,4,5],[5,5,5],
], dtype=int)

C_Cx = np.array([
    [0,0,0,0],[0,1,0,0],[1,0,1,0],[1,1,1,0],[2,0,2,0],[0,1,2,0],
    [3,0,3,0],[0,1,3,0],[4,0,4,0],[0,1,4,0],[5,0,5,0],[0,1,5,0],
    [1,0,0,1],[1,1,0,1],[2,0,1,1],[1,1,1,1],[3,0,2,1],[1,1,2,1],
    [4,0,3,1],[1,1,3,1],[5,0,4,1],[1,1,4,1],[5,0,5,1],[1,1,5,1],
    [2,0,0,2],[2,1,0,2],[3,0,1,2],[2,1,1,2],[4,0,2,2],[2,1,2,2],
    [5,0,3,2],[2,1,3,2],[5,0,4,2],[2,1,4,2],[5,0,5,2],[2,1,5,2],
    [3,0,0,3],[3,1,0,3],[4,0,1,3],[3,1,1,3],[5,0,2,3],[3,1,2,3],
    [5,0,3,3],[3,1,3,3],[5,0,4,3],[3,1,4,3],[5,0,5,3],[3,1,5,3],
    [4,0,0,4],[4,1,0,4],[5,0,1,4],[4,1,1,4],[5,0,2,4],[4,1,2,4],
    [5,0,3,4],[4,1,3,4],[5,0,4,4],[4,1,4,4],[5,0,5,4],[4,1,5,4],
    [5,0,0,5],[5,1,0,5],[5,0,1,5],[5,1,1,5],[5,0,2,5],[5,1,2,5],
    [5,0,3,5],[5,1,3,5],[5,0,4,5],[5,1,4,5],[5,0,5,5],[5,1,5,5],
], dtype=int)

C_Rx = np.array([
    [1,0],[1,1],[2,2],[3,3],[4,4],
], dtype=int)

C_Clx = np.array([
    [0,0,0],[0,1,0],[1,2,0],[1,3,0],[1,4,0],[1,5,0],
    [0,0,1],[0,1,1],[0,2,1],[1,3,1],[1,4,1],[1,5,1],
    [0,0,2],[0,1,2],[0,2,2],[0,3,2],[1,4,2],[1,5,2],
    [0,0,3],[0,1,3],[0,2,3],[0,3,3],[0,4,3],[1,5,3],
    [0,0,4],[0,1,4],[0,2,4],[0,3,4],[0,4,4],[0,5,4],
    [0,0,5],[0,1,5],[0,2,5],[0,3,5],[0,4,5],[0,5,5],
], dtype=int)

C_Zx = np.array([
    [0,0,0],[0,1,0],[1,2,0],[2,3,0],[2,4,0],
    [0,0,1],[1,1,1],[2,2,1],[2,3,1],[3,4,1],
    [1,0,2],[2,1,2],[2,2,2],[3,3,2],[4,4,2],
    [2,0,3],[2,1,3],[3,2,3],[4,3,3],[4,4,3],
    [3,0,4],[3,1,4],[4,2,4],[4,3,4],[4,4,4],
], dtype=int)


def build_fpsobn(n_components: int, n_timesteps: int):
    """Return (varis, cpms, elim_order, query_var) matching FPSOBN.ipynb."""
    varis = {}

    for loc in range(1, n_components + 1):
        for t in range(1, n_timesteps + 1):
            varis[f'Wx{loc}{t}']  = variable.Variable(f'Wx{loc}{t}',  STATES_DAMAGE)
            varis[f'Lx{loc}{t}']  = variable.Variable(f'Lx{loc}{t}',  STATES_DAMAGE)
            varis[f'Tx{loc}{t}']  = variable.Variable(f'Tx{loc}{t}',  STATES_DAMAGE)
            varis[f'Px{loc}{t}']  = variable.Variable(f'Px{loc}{t}',  STATES_BOOL)
            varis[f'Cx{loc}{t}']  = variable.Variable(f'Cx{loc}{t}',  STATES_DAMAGE)
            varis[f'Clx{loc}{t}'] = variable.Variable(f'CLx{loc}{t}', STATES_BOOL)
            varis[f'Rx{loc}{t}']  = variable.Variable(f'Rx{loc}{t}',  STATES_RESIST)
            varis[f'Zx{loc}{t}']  = variable.Variable(f'Zx{loc}{t}',  STATES_AUX)
            varis[f'Vx{loc}{t}']  = variable.Variable(f'Vx{loc}{t}',  STATES_AUX)

    for loc in range(1, n_components + 1):
        varis[f'Cx{loc}0'] = variable.Variable(f'Cx{loc}0', STATES_DAMAGE)

    for t in range(1, n_timesteps + 1):
        varis[f'Ux{t}'] = variable.Variable(f'Ux{t}', STATES_AUX)

    cpms = {}

    for loc in range(1, n_components + 1):
        for t in range(1, n_timesteps + 1):
            cpms[f'Wx{loc}{t}'] = cpm.Cpm(
                variables=[varis[f'Wx{loc}{t}']],
                no_child=1,
                C=np.array([[0],[1],[2],[3],[4],[5]], dtype=int),
                p=np.array([0.017, 0.435, 0.518, 0.03, 0.0, 0.0])
            )
            cpms[f'Lx{loc}{t}'] = cpm.Cpm(
                variables=[varis[f'Lx{loc}{t}']],
                no_child=1,
                C=np.array([[0],[1],[2],[3],[4],[5]], dtype=int),
                p=np.array([0.122, 0.677, 0.198, 0.002, 0.0, 0.0])
            )
            cpms[f'Tx{loc}{t}'] = cpm.Cpm(
                variables=[varis[f'Tx{loc}{t}'], varis[f'Wx{loc}{t}'], varis[f'Lx{loc}{t}']],
                no_child=1,
                C=C_Tx,
                p=np.ones(len(C_Tx))
            )
            cpms[f'Px{loc}{t}'] = cpm.Cpm(
                variables=[varis[f'Px{loc}{t}']],
                no_child=1,
                C=np.array([[0],[1]], dtype=int),
                p=np.array([1.0, 0.0])
            )
            cpms[f'Vx{loc}{t}'] = cpm.Cpm(
                variables=[varis[f'Vx{loc}{t}']],
                no_child=1,
                C=np.array([[0],[1],[2],[3],[4]], dtype=int),
                p=np.array([0.0, 0.006, 0.493, 0.493, 0.006])
            )
            cpms[f'Zx{loc}{t}'] = cpm.Cpm(
                variables=[varis[f'Zx{loc}{t}'], varis[f'Vx{loc}{t}'], varis[f'Ux{t}']],
                no_child=1,
                C=C_Zx,
                p=np.ones(len(C_Zx))
            )
            cpms[f'Rx{loc}{t}'] = cpm.Cpm(
                variables=[varis[f'Rx{loc}{t}'], varis[f'Zx{loc}{t}']],
                no_child=1,
                C=C_Rx,
                p=np.ones(len(C_Rx))
            )
            cpms[f'Cx{loc}{t}'] = cpm.Cpm(
                variables=[varis[f'Cx{loc}{t}'], varis[f'Px{loc}{t}'],
                           varis[f'Cx{loc}{t-1}'], varis[f'Tx{loc}{t}']],
                no_child=1,
                C=C_Cx,
                p=np.ones(len(C_Cx))
            )
            cpms[f'Clx{loc}{t}'] = cpm.Cpm(
                variables=[varis[f'Clx{loc}{t}'], varis[f'Cx{loc}{t}'], varis[f'Rx{loc}{t}']],
                no_child=1,
                C=C_Clx,
                p=np.ones(len(C_Clx))
            )

    for loc in range(1, n_components + 1):
        cpms[f'Cx{loc}0'] = cpm.Cpm(
            variables=[varis[f'Cx{loc}0']],
            no_child=1,
            C=np.array([[0],[1],[2],[3],[4],[5]], dtype=int),
            p=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )

    for t in range(1, n_timesteps + 1):
        cpms[f'Ux{t}'] = cpm.Cpm(
            variables=[varis[f'Ux{t}']],
            no_child=1,
            C=np.array([[0],[1],[2],[3],[4]], dtype=int),
            p=np.array([0.0, 0.006, 0.493, 0.493, 0.006])
        )

    varis_elim_order = []
    for t in range(1, n_timesteps + 1):
        for loc in range(1, n_components + 1):
            varis_elim_order += [varis[f'Wx{loc}{t}'], varis[f'Lx{loc}{t}'], varis[f'Tx{loc}{t}']]
        for loc in range(1, n_components + 1):
            varis_elim_order.append(varis[f'Px{loc}{t}'])
        for loc in range(1, n_components + 1):
            varis_elim_order.append(varis[f'Vx{loc}{t}'])
        varis_elim_order.append(varis[f'Ux{t}'])
        for loc in range(1, n_components + 1):
            varis_elim_order.append(varis[f'Zx{loc}{t}'])
        for loc in range(1, n_components + 1):
            varis_elim_order.append(varis[f'Rx{loc}{t}'])
        for loc in range(1, n_components + 1):
            if not (loc == n_components and t == n_timesteps):
                varis_elim_order.append(varis[f'Clx{loc}{t}'])

    for loc in range(1, n_components + 1):
        varis_elim_order.append(varis[f'Cx{loc}0'])

    for loc in range(1, n_components + 1):
        for t in range(1, n_timesteps + 1):
            varis_elim_order.append(varis[f'Cx{loc}{t}'])

    query_var = varis[f'Clx{n_components}{n_timesteps}']
    return varis, cpms, varis_elim_order, query_var


# ---------------------------------------------------------------------------
# A. Variable creation
# ---------------------------------------------------------------------------

class TestVariables:

    def setup_method(self):
        self.varis, self.cpms, self.elim, self.query = build_fpsobn(2, 2)

    def test_total_variable_count(self):
        n_loc, n_t = 2, 2
        per_loc_per_t = 9   # Wx Lx Tx Px Cx Clx Rx Zx Vx
        cx0_count = n_loc
        ux_count = n_t
        expected = n_loc * n_t * per_loc_per_t + cx0_count + ux_count
        assert len(self.varis) == expected

    def test_damage_variable_state_space(self):
        assert self.varis['Wx11'].values == STATES_DAMAGE
        assert self.varis['Lx11'].values == STATES_DAMAGE
        assert self.varis['Tx11'].values == STATES_DAMAGE
        assert self.varis['Cx11'].values == STATES_DAMAGE
        assert self.varis['Cx10'].values == STATES_DAMAGE

    def test_bool_variable_state_space(self):
        assert self.varis['Px11'].values == STATES_BOOL
        assert self.varis['Clx11'].values == STATES_BOOL

    def test_resist_variable_state_space(self):
        assert self.varis['Rx11'].values == STATES_RESIST

    def test_aux_variable_state_space(self):
        assert self.varis['Vx11'].values == STATES_AUX
        assert self.varis['Zx11'].values == STATES_AUX
        assert self.varis['Ux1'].values  == STATES_AUX

    def test_all_locations_and_timesteps_created(self):
        for loc in range(1, 3):
            for t in range(1, 3):
                for prefix in ['Wx', 'Lx', 'Tx', 'Px', 'Cx', 'Clx', 'Rx', 'Zx', 'Vx']:
                    assert f'{prefix}{loc}{t}' in self.varis

    def test_cx0_created_per_location(self):
        assert 'Cx10' in self.varis
        assert 'Cx20' in self.varis

    def test_ux_created_per_timestep(self):
        assert 'Ux1' in self.varis
        assert 'Ux2' in self.varis
        assert 'Ux0' not in self.varis


# ---------------------------------------------------------------------------
# B. CPM matrices
# ---------------------------------------------------------------------------

class TestCPMMatrices:

    def test_c_tx_shape(self):
        assert C_Tx.shape == (36, 3)

    def test_c_cx_shape(self):
        assert C_Cx.shape == (72, 4)

    def test_c_rx_shape(self):
        assert C_Rx.shape == (5, 2)

    def test_c_clx_shape(self):
        assert C_Clx.shape == (36, 3)

    def test_c_zx_shape(self):
        assert C_Zx.shape == (25, 3)

    def test_c_tx_indices_in_range(self):
        assert C_Tx[:, 0].max() <= 5   # child Tx: 6 damage states
        assert C_Tx[:, 1].max() <= 5   # parent Wx
        assert C_Tx[:, 2].max() <= 5   # parent Lx

    def test_c_cx_indices_in_range(self):
        assert C_Cx[:, 0].max() <= 5   # child Cx
        assert C_Cx[:, 1].max() <= 1   # parent Px (bool)
        assert C_Cx[:, 2].max() <= 5   # parent Cx_prev
        assert C_Cx[:, 3].max() <= 5   # parent Tx

    def test_c_rx_indices_in_range(self):
        assert C_Rx[:, 0].max() <= 5   # child Rx (6 resist states, index 0-5 → but 5 rows cover 0-4)
        assert C_Rx[:, 1].max() <= 4   # parent Zx (5 aux states)

    def test_c_clx_indices_in_range(self):
        assert C_Clx[:, 0].max() <= 1  # child Clx (bool)
        assert C_Clx[:, 1].max() <= 5  # parent Cx
        assert C_Clx[:, 2].max() <= 5  # parent Rx

    def test_c_zx_indices_in_range(self):
        assert C_Zx[:, 0].max() <= 4   # child Zx
        assert C_Zx[:, 1].max() <= 4   # parent Vx
        assert C_Zx[:, 2].max() <= 4   # parent Ux

    def test_c_tx_covers_all_parent_combinations(self):
        # 6 Lx states × 6 Wx states = 36 rows
        parent_pairs = set(map(tuple, C_Tx[:, 1:].tolist()))
        assert len(parent_pairs) == 36

    def test_c_zx_covers_all_parent_combinations(self):
        # 5 Vx × 5 Ux = 25 combinations
        parent_pairs = set(map(tuple, C_Zx[:, 1:].tolist()))
        assert len(parent_pairs) == 25

    def test_c_cx_covers_all_parent_combinations(self):
        # 2 Px × 6 Cx_prev × 6 Tx = 72 rows
        parent_triplets = set(map(tuple, C_Cx[:, 1:].tolist()))
        assert len(parent_triplets) == 72


# ---------------------------------------------------------------------------
# C. Network structure (parent-child wiring)
# ---------------------------------------------------------------------------

class TestNetworkStructure:

    def setup_method(self):
        self.varis, self.cpms, self.elim, self.query = build_fpsobn(2, 2)

    def _cpm_var_names(self, key):
        return [v.name for v in self.cpms[key].variables]

    def test_tx_parents_are_wx_and_lx(self):
        names = self._cpm_var_names('Tx11')
        assert names[0] == 'Tx11'
        assert 'Wx11' in names
        assert 'Lx11' in names

    def test_zx_parents_are_vx_and_ux_for_same_timestep(self):
        names_t1 = self._cpm_var_names('Zx11')
        assert 'Vx11' in names_t1
        assert 'Ux1' in names_t1
        assert 'Ux2' not in names_t1

        names_t2 = self._cpm_var_names('Zx12')
        assert 'Ux2' in names_t2
        assert 'Ux1' not in names_t2

    def test_cx_temporal_chain_references_previous_cx(self):
        # Cx{loc}{t} should reference Cx{loc}{t-1}
        names_t1 = self._cpm_var_names('Cx11')
        assert 'Cx10' in names_t1

        names_t2 = self._cpm_var_names('Cx12')
        assert 'Cx11' in names_t2

    def test_cx_does_not_cross_locations(self):
        # Cx11 chain should not reference Cx20
        names = self._cpm_var_names('Cx11')
        assert 'Cx20' not in names

    def test_clx_parents_are_cx_and_rx(self):
        names = self._cpm_var_names('Clx11')
        assert 'Cx11' in names
        assert 'Rx11' in names

    def test_rx_parent_is_zx(self):
        names = self._cpm_var_names('Rx11')
        assert 'Zx11' in names

    def test_zx_at_loc2_uses_shared_ux(self):
        names_loc1 = self._cpm_var_names('Zx11')
        names_loc2 = self._cpm_var_names('Zx21')
        assert 'Ux1' in names_loc1
        assert 'Ux1' in names_loc2


# ---------------------------------------------------------------------------
# D. Elimination order
# ---------------------------------------------------------------------------

class TestEliminationOrder:

    def setup_method(self):
        self.varis, self.cpms, self.elim, self.query = build_fpsobn(2, 2)

    def test_query_variable_not_in_elim_order(self):
        assert self.query not in self.elim

    def test_no_duplicates_in_elim_order(self):
        assert len(self.elim) == len(set(id(v) for v in self.elim))

    def test_elim_order_length(self):
        n_loc, n_t = 2, 2
        total_vars = n_loc * n_t * 9 + n_loc + n_t
        expected_elim = total_vars - 1  # subtract query
        assert len(self.elim) == expected_elim

    def test_ux_eliminated_within_its_timestep_block(self):
        elim_names = [v.name for v in self.elim]
        ux1_idx = elim_names.index('Ux1')
        ux2_idx = elim_names.index('Ux2')
        # Ux1 comes before Ux2
        assert ux1_idx < ux2_idx

    def test_cx0_eliminated_after_per_timestep_variables(self):
        elim_names = [v.name for v in self.elim]
        cx10_idx = elim_names.index('Cx10')
        ux2_idx  = elim_names.index('Ux2')
        assert cx10_idx > ux2_idx

    def test_cx_temporal_chain_eliminated_last(self):
        elim_names = [v.name for v in self.elim]
        cx10_idx = elim_names.index('Cx10')
        cx11_idx = elim_names.index('Cx11')
        cx12_idx = elim_names.index('Cx12')
        # All Cx0 and Cx{t} come after Ux variables
        assert cx11_idx > cx10_idx
        assert cx12_idx > cx11_idx


# ---------------------------------------------------------------------------
# E. Inference – known result for 2×2 network
# ---------------------------------------------------------------------------

class TestInference:

    def test_marginal_is_valid_probability_distribution(self):
        _, cpms, elim, _ = build_fpsobn(2, 2)
        result = inference.variable_elim(cpms=cpms, var_elim=elim, prod=True)
        p = result.p
        assert np.all(p >= 0.0), "Negative probabilities"
        # variable_elim returns unnormalized factors; check it's close to 1
        assert p.sum() > 0.9, f"Probabilities sum too low: {p.sum()}"
        assert p.sum() < 1.05, f"Probabilities sum too high: {p.sum()}"

    def test_marginal_has_two_states(self):
        _, cpms, elim, _ = build_fpsobn(2, 2)
        result = inference.variable_elim(cpms=cpms, var_elim=elim, prod=True)
        assert len(result.p) == 2

    def test_known_result_2x2(self):
        # Verified from notebook: P(CLx22=False)≈0.3854, P(CLx22=True)≈0.5988
        _, cpms, elim, _ = build_fpsobn(2, 2)
        result = inference.variable_elim(cpms=cpms, var_elim=elim, prod=True)
        p = result.p
        assert abs(p[0] - 0.3854) < 0.001, f"P(False)={p[0]:.4f}, expected ~0.3854"
        assert abs(p[1] - 0.5988) < 0.001, f"P(True)={p[1]:.4f}, expected ~0.5988"

    def test_crack_probability_is_majority(self):
        # With this damage model, crack is more likely than not at t=2
        _, cpms, elim, _ = build_fpsobn(2, 2)
        result = inference.variable_elim(cpms=cpms, var_elim=elim, prod=True)
        p = result.p
        assert p[1] > p[0], "Expected P(crack=True) > P(crack=False)"


# ---------------------------------------------------------------------------
# F. Temporal chain correctness
# ---------------------------------------------------------------------------

class TestTemporalChain:

    def test_cx0_prior_is_no_damage(self):
        varis, cpms, _, _ = build_fpsobn(2, 2)
        for loc in range(1, 3):
            p = cpms[f'Cx{loc}0'].p
            assert p[0] == pytest.approx(1.0), f"Cx{loc}0 state 0 probability should be 1"
            assert all(p[i] == pytest.approx(0.0) for i in range(1, 6))

    def test_cx_at_t1_depends_on_cx0_not_cx_other_t(self):
        varis, cpms, _, _ = build_fpsobn(2, 3)
        for loc in range(1, 3):
            var_names = [v.name for v in cpms[f'Cx{loc}1'].variables]
            assert f'Cx{loc}0' in var_names
            assert f'Cx{loc}2' not in var_names
            assert f'Cx{loc}3' not in var_names

    def test_cx_at_t2_depends_on_cx_t1(self):
        varis, cpms, _, _ = build_fpsobn(2, 2)
        for loc in range(1, 3):
            var_names = [v.name for v in cpms[f'Cx{loc}2'].variables]
            assert f'Cx{loc}1' in var_names

    def test_px_prior_means_no_protection(self):
        # Px is permanently False (p=[1,0]) meaning no protection applied
        varis, cpms, _, _ = build_fpsobn(2, 2)
        for loc in range(1, 3):
            for t in range(1, 3):
                p = cpms[f'Px{loc}{t}'].p
                assert p[0] == pytest.approx(1.0)
                assert p[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# G. Ux per time-step
# ---------------------------------------------------------------------------

class TestUxPerTimestep:

    def test_distinct_ux_objects_per_timestep(self):
        varis, _, _, _ = build_fpsobn(2, 3)
        assert varis['Ux1'] is not varis['Ux2']
        assert varis['Ux2'] is not varis['Ux3']

    def test_zx_at_t1_references_ux1_not_ux2(self):
        varis, cpms, _, _ = build_fpsobn(2, 2)
        for loc in range(1, 3):
            zx_vars = cpms[f'Zx{loc}1'].variables
            assert varis['Ux1'] in zx_vars
            assert varis['Ux2'] not in zx_vars

    def test_zx_at_t2_references_ux2_not_ux1(self):
        varis, cpms, _, _ = build_fpsobn(2, 2)
        for loc in range(1, 3):
            zx_vars = cpms[f'Zx{loc}2'].variables
            assert varis['Ux2'] in zx_vars
            assert varis['Ux1'] not in zx_vars

    def test_ux_count_equals_n_timesteps(self):
        for n_t in [1, 2, 3]:
            varis, _, _, _ = build_fpsobn(2, n_t)
            ux_keys = [k for k in varis if k.startswith('Ux')]
            assert len(ux_keys) == n_t

    def test_ux_prior_matches_known_values(self):
        # Prior: [0.0, 0.006, 0.493, 0.493, 0.006] for states [-0.5,-0.25,0.0,0.25,0.5]
        varis, cpms, _, _ = build_fpsobn(2, 2)
        expected = np.array([0.0, 0.006, 0.493, 0.493, 0.006])
        for t in range(1, 3):
            p = cpms[f'Ux{t}'].p.flatten()
            np.testing.assert_allclose(p, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# H. Scalability
# ---------------------------------------------------------------------------

class TestScalability:

    @pytest.mark.parametrize("n_loc,n_t", [(1, 1), (2, 1), (1, 2), (3, 2)])
    def test_build_completes(self, n_loc, n_t):
        varis, cpms, elim, query = build_fpsobn(n_loc, n_t)
        assert len(varis) > 0
        assert len(cpms) > 0
        assert len(elim) > 0

    @pytest.mark.parametrize("n_loc,n_t", [(1, 1), (2, 1), (1, 2)])
    def test_inference_completes(self, n_loc, n_t):
        _, cpms, elim, _ = build_fpsobn(n_loc, n_t)
        result = inference.variable_elim(cpms=cpms, var_elim=elim, prod=True)
        p = result.p
        assert np.all(p >= 0.0)
        assert p.sum() > 0.9

    def test_1x1_query_is_clx11(self):
        varis, cpms, elim, query = build_fpsobn(1, 1)
        assert query is varis['Clx11']

    def test_query_is_always_last_loc_last_timestep(self):
        for n_loc, n_t in [(1, 1), (2, 2), (3, 2)]:
            varis, _, _, query = build_fpsobn(n_loc, n_t)
            assert query is varis[f'Clx{n_loc}{n_t}']
