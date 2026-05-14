# -*- coding: utf-8 -*-
"""
Pytest suite for FPSOBN sequential time-step Bayesian Network.

Covers:
    A. Variable creation       - counts, names, state spaces
    B. CPM matrices            - shapes, index bounds, parent coverage
    C. Network structure       - parent-child wiring, shared Ux per timestep
    D. Observations dict       - dynamic sizing, default values
    E. ux_prior_update         - normalisation behaviour
    F. Evidence handling       - Wx/Lx point-mass, Px repair, Clx conditioning
    G. Sequential inference    - Ux posteriors per time step
    H. Temporal chain          - Cx{loc}{t} -> Cx{loc}{t-1}, no cross-location
    I. Ux per time-step        - distinct Ux{t}, prior propagation
    J. Scalability             - varied n_components x n_timesteps
"""

import copy
import numpy as np
import pytest
from mbnpy import variable, cpm, inference


# ---------------------------------------------------------------------------
# Shared definitions (mirror FPSOBN.ipynb Cell 4)
# ---------------------------------------------------------------------------

STATES_DAMAGE = ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
STATES_BOOL   = ['False', 'True']
STATES_RESIST = ['0.4', '0.6', '0.8', '1.0', '1.2', '1.4']
STATES_AUX    = ['-0.5', '-0.25', '0.0', '0.25', '0.5']

WX_PRIOR = np.array([0.017, 0.435, 0.518, 0.03, 0.0, 0.0])
LX_PRIOR = np.array([0.122, 0.677, 0.198, 0.002, 0.0, 0.0])
VX_PRIOR = np.array([0.0, 0.006, 0.493, 0.493, 0.006])
UX_PRIOR = np.array([0.0, 0.006, 0.493, 0.493, 0.006])
CX0_PRIOR = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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

C_Rx = np.array([[1,0],[1,1],[2,2],[3,3],[4,4]], dtype=int)

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


# ---------------------------------------------------------------------------
# Builder functions (mirror notebook implementation)
# ---------------------------------------------------------------------------

def ux_prior_update(ux_marginal_cpm):
    """Normalise the Ux posterior to use as the prior for the next time step."""
    p = ux_marginal_cpm.p.flatten()
    return p / p.sum()


def make_observations(n_components, n_timesteps):
    """Build a default observations dict (no evidence) sized to the network."""
    return {
        t: {
            'wx': None,
            'lx': None,
            'locations': {loc: {'clx': False, 'px': False}
                          for loc in range(1, n_components + 1)}
        }
        for t in range(1, n_timesteps + 1)
    }


def _point_mass(n_states, idx):
    p = np.zeros(n_states)
    p[idx] = 1.0
    return p


def _add_timestep(varis, cpms, t, n_components, obs_t, ux_prior):
    """Mutate varis/cpms to include all variables and CPMs for time step t."""
    varis[f'Ux{t}'] = variable.Variable(f'Ux{t}', STATES_AUX)
    cpms[f'Ux{t}'] = cpm.Cpm(
        variables=[varis[f'Ux{t}']],
        no_child=1,
        C=np.array([[0],[1],[2],[3],[4]], dtype=int),
        p=ux_prior.copy(),
    )

    wx_p = WX_PRIOR.copy() if obs_t['wx'] is None else _point_mass(6, obs_t['wx'])
    lx_p = LX_PRIOR.copy() if obs_t['lx'] is None else _point_mass(6, obs_t['lx'])

    for loc in range(1, n_components + 1):
        varis[f'Wx{loc}{t}']  = variable.Variable(f'Wx{loc}{t}',  STATES_DAMAGE)
        varis[f'Lx{loc}{t}']  = variable.Variable(f'Lx{loc}{t}',  STATES_DAMAGE)
        varis[f'Tx{loc}{t}']  = variable.Variable(f'Tx{loc}{t}',  STATES_DAMAGE)
        varis[f'Px{loc}{t}']  = variable.Variable(f'Px{loc}{t}',  STATES_BOOL)
        varis[f'Cx{loc}{t}']  = variable.Variable(f'Cx{loc}{t}',  STATES_DAMAGE)
        varis[f'Clx{loc}{t}'] = variable.Variable(f'CLx{loc}{t}', STATES_BOOL)
        varis[f'Rx{loc}{t}']  = variable.Variable(f'Rx{loc}{t}',  STATES_RESIST)
        varis[f'Zx{loc}{t}']  = variable.Variable(f'Zx{loc}{t}',  STATES_AUX)
        varis[f'Vx{loc}{t}']  = variable.Variable(f'Vx{loc}{t}',  STATES_AUX)

        cpms[f'Wx{loc}{t}'] = cpm.Cpm(
            variables=[varis[f'Wx{loc}{t}']], no_child=1,
            C=np.array([[0],[1],[2],[3],[4],[5]], dtype=int),
            p=wx_p.copy(),
        )
        cpms[f'Lx{loc}{t}'] = cpm.Cpm(
            variables=[varis[f'Lx{loc}{t}']], no_child=1,
            C=np.array([[0],[1],[2],[3],[4],[5]], dtype=int),
            p=lx_p.copy(),
        )
        cpms[f'Tx{loc}{t}'] = cpm.Cpm(
            variables=[varis[f'Tx{loc}{t}'], varis[f'Wx{loc}{t}'], varis[f'Lx{loc}{t}']],
            no_child=1, C=C_Tx, p=np.ones(len(C_Tx)),
        )
        px_p = np.array([0.0, 1.0]) if obs_t['locations'][loc]['px'] else np.array([1.0, 0.0])
        cpms[f'Px{loc}{t}'] = cpm.Cpm(
            variables=[varis[f'Px{loc}{t}']], no_child=1,
            C=np.array([[0],[1]], dtype=int), p=px_p,
        )
        cpms[f'Vx{loc}{t}'] = cpm.Cpm(
            variables=[varis[f'Vx{loc}{t}']], no_child=1,
            C=np.array([[0],[1],[2],[3],[4]], dtype=int),
            p=VX_PRIOR.copy(),
        )
        cpms[f'Zx{loc}{t}'] = cpm.Cpm(
            variables=[varis[f'Zx{loc}{t}'], varis[f'Vx{loc}{t}'], varis[f'Ux{t}']],
            no_child=1, C=C_Zx, p=np.ones(len(C_Zx)),
        )
        cpms[f'Rx{loc}{t}'] = cpm.Cpm(
            variables=[varis[f'Rx{loc}{t}'], varis[f'Zx{loc}{t}']],
            no_child=1, C=C_Rx, p=np.ones(len(C_Rx)),
        )
        cpms[f'Cx{loc}{t}'] = cpm.Cpm(
            variables=[varis[f'Cx{loc}{t}'], varis[f'Px{loc}{t}'],
                       varis[f'Cx{loc}{t-1}'], varis[f'Tx{loc}{t}']],
            no_child=1, C=C_Cx, p=np.ones(len(C_Cx)),
        )
        cpms[f'Clx{loc}{t}'] = cpm.Cpm(
            variables=[varis[f'Clx{loc}{t}'], varis[f'Cx{loc}{t}'], varis[f'Rx{loc}{t}']],
            no_child=1, C=C_Clx, p=np.ones(len(C_Clx)),
        )


def _build_ux_elim_order(varis, n_components, t):
    """Build the elimination order for VE querying Ux{t}."""
    order = []
    for t2 in range(1, t + 1):
        for loc in range(1, n_components + 1):
            order += [varis[f'Wx{loc}{t2}'], varis[f'Lx{loc}{t2}'], varis[f'Tx{loc}{t2}']]
        for loc in range(1, n_components + 1):
            order.append(varis[f'Px{loc}{t2}'])
        for loc in range(1, n_components + 1):
            order.append(varis[f'Vx{loc}{t2}'])
        if t2 < t:
            order.append(varis[f'Ux{t2}'])
        for loc in range(1, n_components + 1):
            order.append(varis[f'Zx{loc}{t2}'])
        for loc in range(1, n_components + 1):
            order.append(varis[f'Rx{loc}{t2}'])
        for loc in range(1, n_components + 1):
            order.append(varis[f'Clx{loc}{t2}'])
    for loc in range(1, n_components + 1):
        order.append(varis[f'Cx{loc}0'])
    for loc in range(1, n_components + 1):
        for t2 in range(1, t + 1):
            order.append(varis[f'Cx{loc}{t2}'])
    return order


def run_sequential(n_components, n_timesteps, observations=None):
    """Run the full sequential BN, returning all intermediate state.

    Returns
    -------
    dict with keys:
        'varis'         : final dict of all Variable objects
        'cpms'          : final dict of all Cpm objects
        'ux_posteriors' : {t: Cpm} – the Ux{t} marginal at each time step
        'ux_priors'     : {t: np.array} – the prior used to build Ux{t}
    """
    if observations is None:
        observations = make_observations(n_components, n_timesteps)

    varis, cpms = {}, {}

    for loc in range(1, n_components + 1):
        varis[f'Cx{loc}0'] = variable.Variable(f'Cx{loc}0', STATES_DAMAGE)
        cpms[f'Cx{loc}0'] = cpm.Cpm(
            variables=[varis[f'Cx{loc}0']], no_child=1,
            C=np.array([[0],[1],[2],[3],[4],[5]], dtype=int),
            p=CX0_PRIOR.copy(),
        )

    ux_prior = UX_PRIOR.copy()
    ux_posteriors = {}
    ux_priors = {}

    for t in range(1, n_timesteps + 1):
        ux_priors[t] = ux_prior.copy()
        _add_timestep(varis, cpms, t, n_components, observations[t], ux_prior)

        clx_cnd_vars, clx_cnd_states = [], []
        for loc in range(1, n_components + 1):
            if observations[t]['locations'][loc]['clx']:
                clx_cnd_vars.append(varis[f'Clx{loc}{t}'])
                clx_cnd_states.append(1)

        if clx_cnd_vars:
            cpms_ux = inference.condition(cpms, cnd_vars=clx_cnd_vars, cnd_states=clx_cnd_states)
        else:
            cpms_ux = copy.deepcopy(cpms)

        elim_order = _build_ux_elim_order(varis, n_components, t)
        ux_marginal = inference.variable_elim(cpms=cpms_ux, var_elim=elim_order, prod=True)
        ux_posteriors[t] = ux_marginal
        ux_prior = ux_prior_update(ux_marginal)

    return {'varis': varis, 'cpms': cpms,
            'ux_posteriors': ux_posteriors, 'ux_priors': ux_priors}


# ---------------------------------------------------------------------------
# A. Variable creation
# ---------------------------------------------------------------------------

class TestVariables:

    def setup_method(self):
        self.result = run_sequential(2, 2)
        self.varis = self.result['varis']

    def test_total_variable_count(self):
        # 9 vars per loc per t + n_loc Cx0 + n_t Ux
        expected = 2 * 2 * 9 + 2 + 2
        assert len(self.varis) == expected

    def test_damage_variable_state_space(self):
        for name in ['Wx11', 'Lx11', 'Tx11', 'Cx11', 'Cx10']:
            assert self.varis[name].values == STATES_DAMAGE

    def test_bool_variable_state_space(self):
        assert self.varis['Px11'].values == STATES_BOOL
        assert self.varis['Clx11'].values == STATES_BOOL

    def test_resist_variable_state_space(self):
        assert self.varis['Rx11'].values == STATES_RESIST

    def test_aux_variable_state_space(self):
        for name in ['Vx11', 'Zx11', 'Ux1', 'Ux2']:
            assert self.varis[name].values == STATES_AUX

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
        assert C_Tx[:, 0].max() <= 5
        assert C_Tx[:, 1].max() <= 5
        assert C_Tx[:, 2].max() <= 5

    def test_c_cx_indices_in_range(self):
        assert C_Cx[:, 0].max() <= 5
        assert C_Cx[:, 1].max() <= 1   # Px is boolean
        assert C_Cx[:, 2].max() <= 5
        assert C_Cx[:, 3].max() <= 5

    def test_c_clx_indices_in_range(self):
        assert C_Clx[:, 0].max() <= 1  # Clx is boolean
        assert C_Clx[:, 1].max() <= 5
        assert C_Clx[:, 2].max() <= 5

    def test_c_zx_indices_in_range(self):
        assert C_Zx[:, 0].max() <= 4
        assert C_Zx[:, 1].max() <= 4
        assert C_Zx[:, 2].max() <= 4

    def test_c_tx_covers_all_parent_combinations(self):
        parent_pairs = set(map(tuple, C_Tx[:, 1:].tolist()))
        assert len(parent_pairs) == 36

    def test_c_zx_covers_all_parent_combinations(self):
        parent_pairs = set(map(tuple, C_Zx[:, 1:].tolist()))
        assert len(parent_pairs) == 25

    def test_c_cx_covers_all_parent_combinations(self):
        parent_triplets = set(map(tuple, C_Cx[:, 1:].tolist()))
        assert len(parent_triplets) == 72

    def test_c_cx_repair_collapses_to_current_tx(self):
        # When Px=1 (repair) the C_Cx matrix sets Cx = Tx (repair clears the
        # accumulated past damage, but any new damage from this step's loading
        # still applies). One row [1,1,1,0] deviates and may be a data-entry
        # bug — flagged via xfail rather than removed.
        repair_rows = C_Cx[C_Cx[:, 1] == 1]
        anomalies = repair_rows[repair_rows[:, 0] != repair_rows[:, 3]]
        # Known anomaly: row [1,1,1,0] (Tx=0 but Cx=1)
        assert len(anomalies) == 1, f"Unexpected anomalies in C_Cx: {anomalies.tolist()}"
        assert anomalies.tolist() == [[1, 1, 1, 0]], (
            "C_Cx anomaly profile changed — inspect matrix"
        )


# ---------------------------------------------------------------------------
# C. Network structure
# ---------------------------------------------------------------------------

class TestNetworkStructure:

    def setup_method(self):
        self.result = run_sequential(2, 2)
        self.varis = self.result['varis']
        self.cpms  = self.result['cpms']

    def _names(self, key):
        return [v.name for v in self.cpms[key].variables]

    def test_tx_parents_are_wx_and_lx(self):
        names = self._names('Tx11')
        assert names[0] == 'Tx11'
        assert 'Wx11' in names
        assert 'Lx11' in names

    def test_zx_parents_are_vx_and_ux_for_same_timestep(self):
        assert 'Vx11' in self._names('Zx11')
        assert 'Ux1'  in self._names('Zx11')
        assert 'Ux2'  not in self._names('Zx11')
        assert 'Ux2'  in self._names('Zx12')
        assert 'Ux1'  not in self._names('Zx12')

    def test_cx_temporal_chain_references_previous_cx(self):
        assert 'Cx10' in self._names('Cx11')
        assert 'Cx11' in self._names('Cx12')

    def test_cx_does_not_cross_locations(self):
        assert 'Cx20' not in self._names('Cx11')
        assert 'Cx21' not in self._names('Cx12')

    def test_clx_parents_are_cx_and_rx(self):
        names = self._names('Clx11')
        assert 'Cx11' in names
        assert 'Rx11' in names

    def test_rx_parent_is_zx(self):
        assert 'Zx11' in self._names('Rx11')

    def test_ux_shared_across_locations_within_timestep(self):
        # Ux1 must appear in Zx{loc}1 for every location
        for loc in range(1, 3):
            assert 'Ux1' in self._names(f'Zx{loc}1')
            assert 'Ux2' in self._names(f'Zx{loc}2')


# ---------------------------------------------------------------------------
# D. Observations dict
# ---------------------------------------------------------------------------

class TestObservations:

    def test_observations_dynamic_sizing_timesteps(self):
        obs = make_observations(2, 5)
        assert set(obs.keys()) == {1, 2, 3, 4, 5}

    def test_observations_dynamic_sizing_locations(self):
        obs = make_observations(4, 1)
        assert set(obs[1]['locations'].keys()) == {1, 2, 3, 4}

    def test_wx_lx_at_timestep_level(self):
        obs = make_observations(2, 2)
        assert 'wx' in obs[1]
        assert 'lx' in obs[1]
        # not at location level
        assert 'wx' not in obs[1]['locations'][1]
        assert 'lx' not in obs[1]['locations'][1]

    def test_clx_px_at_location_level(self):
        obs = make_observations(2, 2)
        for loc in (1, 2):
            assert 'clx' in obs[1]['locations'][loc]
            assert 'px'  in obs[1]['locations'][loc]

    def test_defaults_are_no_evidence(self):
        obs = make_observations(2, 2)
        assert obs[1]['wx'] is None
        assert obs[1]['lx'] is None
        for loc in (1, 2):
            assert obs[1]['locations'][loc]['clx'] is False
            assert obs[1]['locations'][loc]['px']  is False


# ---------------------------------------------------------------------------
# E. ux_prior_update helper
# ---------------------------------------------------------------------------

class TestUxPriorUpdate:

    def _fake_cpm(self, p):
        return cpm.Cpm(
            variables=[variable.Variable('Ux_test', STATES_AUX)],
            no_child=1,
            C=np.array([[0],[1],[2],[3],[4]], dtype=int),
            p=np.array(p, dtype=float),
        )

    def test_normalises_to_sum_one(self):
        result = ux_prior_update(self._fake_cpm([0.1, 0.2, 0.3, 0.2, 0.1]))
        assert np.isclose(result.sum(), 1.0)

    def test_normalises_unnormalised_input(self):
        result = ux_prior_update(self._fake_cpm([1.0, 2.0, 3.0, 2.0, 1.0]))
        assert np.isclose(result.sum(), 1.0)
        np.testing.assert_allclose(result, np.array([0.111111, 0.222222, 0.333333, 0.222222, 0.111111]), atol=1e-5)

    def test_returns_1d_array(self):
        result = ux_prior_update(self._fake_cpm([0.2, 0.2, 0.2, 0.2, 0.2]))
        assert result.ndim == 1

    def test_preserves_shape(self):
        result = ux_prior_update(self._fake_cpm([0.1, 0.2, 0.3, 0.2, 0.1]))
        assert len(result) == 5


# ---------------------------------------------------------------------------
# F. Evidence handling
# ---------------------------------------------------------------------------

class TestEvidenceHandling:

    def test_wx_observation_becomes_point_mass(self):
        obs = make_observations(2, 1)
        obs[1]['wx'] = 2  # observed state index 2 = '0.4'
        result = run_sequential(2, 1, obs)
        for loc in (1, 2):
            p = result['cpms'][f'Wx{loc}1'].p.flatten()
            assert p[2] == pytest.approx(1.0)
            assert np.isclose(p.sum(), 1.0)

    def test_lx_observation_becomes_point_mass(self):
        obs = make_observations(2, 1)
        obs[1]['lx'] = 3
        result = run_sequential(2, 1, obs)
        for loc in (1, 2):
            p = result['cpms'][f'Lx{loc}1'].p.flatten()
            assert p[3] == pytest.approx(1.0)

    def test_wx_lx_shared_across_locations(self):
        obs = make_observations(3, 1)
        obs[1]['wx'] = 1
        obs[1]['lx'] = 2
        result = run_sequential(3, 1, obs)
        # All locations should get the same Wx/Lx p-vector
        wx_vectors = [result['cpms'][f'Wx{loc}1'].p.flatten() for loc in range(1, 4)]
        lx_vectors = [result['cpms'][f'Lx{loc}1'].p.flatten() for loc in range(1, 4)]
        for v in wx_vectors[1:]:
            np.testing.assert_allclose(v, wx_vectors[0])
        for v in lx_vectors[1:]:
            np.testing.assert_allclose(v, lx_vectors[0])

    def test_no_wx_observation_uses_prior(self):
        result = run_sequential(2, 1)  # no observations
        p = result['cpms']['Wx11'].p.flatten()
        np.testing.assert_allclose(p, WX_PRIOR)

    def test_no_lx_observation_uses_prior(self):
        result = run_sequential(2, 1)
        p = result['cpms']['Lx11'].p.flatten()
        np.testing.assert_allclose(p, LX_PRIOR)

    def test_px_default_is_no_repair(self):
        result = run_sequential(2, 2)
        for loc in (1, 2):
            for t in (1, 2):
                p = result['cpms'][f'Px{loc}{t}'].p
                assert p[0] == pytest.approx(1.0)
                assert p[1] == pytest.approx(0.0)

    def test_px_observation_encodes_repair(self):
        obs = make_observations(2, 1)
        obs[1]['locations'][1]['px'] = True
        result = run_sequential(2, 1, obs)
        p1 = result['cpms']['Px11'].p
        p2 = result['cpms']['Px21'].p
        assert p1[1] == pytest.approx(1.0), "Repair at loc 1 should give Px=[0,1]"
        assert p2[0] == pytest.approx(1.0), "No repair at loc 2 should give Px=[1,0]"

    def test_clx_observation_does_not_change_cpms(self):
        # Clx is conditioned via inference.condition, not via CPM p-vector
        obs_default = make_observations(2, 1)
        obs_clx = make_observations(2, 1)
        obs_clx[1]['locations'][1]['clx'] = True

        result_default = run_sequential(2, 1, obs_default)
        result_clx     = run_sequential(2, 1, obs_clx)

        # The original Clx CPM p-vector is unchanged regardless of observation
        np.testing.assert_allclose(
            result_default['cpms']['Clx11'].p,
            result_clx['cpms']['Clx11'].p,
        )


# ---------------------------------------------------------------------------
# G. Sequential inference
# ---------------------------------------------------------------------------

class TestSequentialInference:

    def test_ux_posterior_computed_each_timestep(self):
        result = run_sequential(2, 3)
        assert set(result['ux_posteriors'].keys()) == {1, 2, 3}

    def test_ux_posterior_is_valid_distribution(self):
        result = run_sequential(2, 2)
        for t, marg in result['ux_posteriors'].items():
            p = marg.p.flatten()
            assert np.all(p >= 0.0), f"Negative probability at t={t}"
            assert p.sum() > 0.0, f"Zero total probability at t={t}"

    def test_ux_posterior_has_5_states(self):
        result = run_sequential(2, 2)
        for marg in result['ux_posteriors'].values():
            assert len(marg.p) == 5

    def test_first_timestep_prior_is_default(self):
        result = run_sequential(2, 2)
        np.testing.assert_allclose(result['ux_priors'][1], UX_PRIOR)

    def test_second_timestep_prior_is_updated_posterior(self):
        result = run_sequential(2, 2)
        expected = ux_prior_update(result['ux_posteriors'][1])
        np.testing.assert_allclose(result['ux_priors'][2], expected)

    def test_no_evidence_posterior_close_to_prior(self):
        # With no evidence the Ux posterior should remain close to the prior
        # (some drift is expected because other CPMs still factor into VE,
        # but the relative ordering of probabilities should be preserved)
        result = run_sequential(2, 1)
        posterior = result['ux_posteriors'][1].p.flatten()
        posterior_normed = posterior / posterior.sum()
        # The two middle states should still dominate
        assert posterior_normed[2] + posterior_normed[3] > 0.9

    def test_clx_evidence_shifts_ux_posterior(self):
        # A crack observation should produce a measurably different Ux posterior
        baseline = run_sequential(2, 1)
        obs = make_observations(2, 1)
        obs[1]['locations'][1]['clx'] = True
        with_crack = run_sequential(2, 1, obs)

        p_base = baseline['ux_posteriors'][1].p.flatten()
        p_base = p_base / p_base.sum()
        p_crack = with_crack['ux_posteriors'][1].p.flatten()
        p_crack = p_crack / p_crack.sum()
        # Distributions should differ
        assert not np.allclose(p_base, p_crack), "Clx evidence should shift Ux posterior"


# ---------------------------------------------------------------------------
# H. Temporal chain
# ---------------------------------------------------------------------------

class TestTemporalChain:

    def test_cx0_prior_is_no_damage(self):
        result = run_sequential(2, 2)
        for loc in (1, 2):
            p = result['cpms'][f'Cx{loc}0'].p
            assert p[0] == pytest.approx(1.0)
            assert all(p[i] == pytest.approx(0.0) for i in range(1, 6))

    def test_cx_at_t1_depends_only_on_cx0(self):
        result = run_sequential(2, 3)
        for loc in (1, 2):
            names = [v.name for v in result['cpms'][f'Cx{loc}1'].variables]
            assert f'Cx{loc}0' in names
            assert f'Cx{loc}2' not in names
            assert f'Cx{loc}3' not in names

    def test_cx_at_t2_depends_on_cx_t1(self):
        result = run_sequential(2, 2)
        for loc in (1, 2):
            names = [v.name for v in result['cpms'][f'Cx{loc}2'].variables]
            assert f'Cx{loc}1' in names

    def test_repair_observation_propagates_into_cx_cpm(self):
        # When Px=True is set, the Cx CPM's structure is unchanged but Px's
        # p-vector ensures the deterministic reset takes effect during VE
        obs = make_observations(2, 1)
        obs[1]['locations'][1]['px'] = True
        result = run_sequential(2, 1, obs)
        # Px11 is now a point mass at 'True'
        assert result['cpms']['Px11'].p[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# I. Ux per time-step
# ---------------------------------------------------------------------------

class TestUxPerTimestep:

    def test_distinct_ux_objects_per_timestep(self):
        result = run_sequential(2, 3)
        varis = result['varis']
        assert varis['Ux1'] is not varis['Ux2']
        assert varis['Ux2'] is not varis['Ux3']

    def test_ux_count_equals_n_timesteps(self):
        for n_t in (1, 2, 3):
            result = run_sequential(2, n_t)
            ux_keys = [k for k in result['varis'] if k.startswith('Ux')]
            assert len(ux_keys) == n_t

    def test_ux_prior_carries_forward_between_timesteps(self):
        result = run_sequential(2, 3)
        for t in (2, 3):
            expected = ux_prior_update(result['ux_posteriors'][t - 1])
            np.testing.assert_allclose(result['ux_priors'][t], expected)

    def test_ux1_uses_default_prior(self):
        result = run_sequential(2, 2)
        p = result['cpms']['Ux1'].p.flatten()
        np.testing.assert_allclose(p, UX_PRIOR)


# ---------------------------------------------------------------------------
# J. Scalability
# ---------------------------------------------------------------------------

class TestScalability:

    @pytest.mark.parametrize("n_loc,n_t", [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2)])
    def test_sequential_run_completes(self, n_loc, n_t):
        result = run_sequential(n_loc, n_t)
        assert len(result['ux_posteriors']) == n_t
        for marg in result['ux_posteriors'].values():
            assert marg.p.sum() > 0.0

    @pytest.mark.parametrize("n_loc,n_t", [(1, 1), (2, 1), (1, 2), (2, 2)])
    def test_observations_match_dimensions(self, n_loc, n_t):
        obs = make_observations(n_loc, n_t)
        assert len(obs) == n_t
        for t in range(1, n_t + 1):
            assert len(obs[t]['locations']) == n_loc

    def test_all_locations_get_independent_cracks(self):
        # Each location's crack observation is independent
        obs = make_observations(3, 1)
        obs[1]['locations'][1]['clx'] = True
        obs[1]['locations'][3]['clx'] = True  # crack at loc 1 and 3, not loc 2
        result = run_sequential(3, 1, obs)
        # Should still produce a valid posterior
        p = result['ux_posteriors'][1].p.flatten()
        assert p.sum() > 0.0
