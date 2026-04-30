# -*- coding: utf-8 -*-
"""
Generalised Temporal Bayesian Network Builder (v2).

Extends the original cumulative damage BN with three new variable types:
    - Zx{i}  : auxiliary variable, conditioned on Vx{i} and Ux
    - Vx{i}  : per-component random variable
    - Ux     : single shared random variable (one instance regardless of n)

Rx{i} is now conditioned on Zx{i} (deterministic mapping).
"""

import numpy as np
from mbnpy import variable, cpm, inference


# ---------------------------------------------------------------------------
# Default CPM data
# ---------------------------------------------------------------------------

DEFAULT_WX_PROBS = np.array([0.017, 0.435, 0.518, 0.03, 0.0, 0.0])
DEFAULT_LX_PROBS = np.array([0.122, 0.677, 0.198, 0.002, 0.0, 0.0])
DEFAULT_PX_PROBS = np.array([1.0, 0.0])
DEFAULT_VX_PROBS = np.array([0.0, 0.006, 0.493, 0.493, 0.006])
DEFAULT_UX_PROBS = np.array([0.0, 0.006, 0.493, 0.493, 0.006])

DAMAGE_STATES = ['0', '0.2', '0.4', '0.6', '0.8', '1.0']
REPAIR_STATES = ['False', 'True']
RESISTANCE_STATES = ['0.4', '0.6', '0.8', '1.0', '1.2', '1.4']
NOISE_STATES = ['-0.5', '-0.25', '0.0', '0.25', '0.5']

# Tx CPM table (Tx, Wx, Lx)
TX_C_MATRIX = np.array([
    [0,0,0],[1,1,0],[2,2,0],[3,3,0],[4,4,0],[5,5,0],
    [1,0,1],[2,1,1],[3,2,1],[4,3,1],[5,4,1],[5,5,1],
    [0,0,2],[3,1,2],[4,2,2],[5,3,2],[5,4,2],[5,5,2],
    [3,0,3],[4,1,3],[5,2,3],[5,3,3],[5,4,3],[5,5,3],
    [4,0,4],[5,1,4],[5,2,4],[5,3,4],[5,4,4],[5,5,4],
    [5,0,5],[2,1,5],[5,2,5],[5,3,5],[5,4,5],[5,5,5],
], dtype=int)

# Cx CPM table (Cx, Px, Cx_prev, Tx)
CX_C_MATRIX = np.array([
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

# Clx CPM table (Clx, Cx, Rx)
CLX_C_MATRIX = np.array([
    [0,0,0],[0,1,0],[1,2,0],[1,3,0],[1,4,0],[1,5,0],
    [0,0,1],[0,1,1],[0,2,1],[1,3,1],[1,4,1],[1,5,1],
    [0,0,2],[0,1,2],[0,2,2],[0,3,2],[1,4,2],[1,5,2],
    [0,0,3],[0,1,3],[0,2,3],[0,3,3],[0,4,3],[1,5,3],
    [0,0,4],[0,1,4],[0,2,4],[0,3,4],[0,4,4],[0,5,4],
    [0,0,5],[0,1,5],[0,2,5],[0,3,5],[0,4,5],[0,5,5],
], dtype=int)

# Rx CPM table (Rx, Zx) — deterministic; only Zx states 0..4 are reachable,
# Rx states 0 and 5 are never activated by design.
RX_C_MATRIX = np.array([
    [1,0],[1,1],[2,2],[3,3],[4,4],
], dtype=int)

# Zx CPM table (Zx, Vx, Ux)
ZX_C_MATRIX = np.array([
    [0,0,0],[0,1,0],[1,2,0],[2,3,0],[2,4,0],
    [0,0,1],[1,1,1],[2,2,1],[2,3,1],[3,4,1],
    [1,0,2],[2,1,2],[2,2,2],[3,3,2],[4,4,2],
    [2,0,3],[2,1,3],[3,2,3],[4,3,3],[4,4,3],
    [3,0,4],[3,1,4],[4,2,4],[4,3,4],[4,4,4],
], dtype=int)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_cx0_probs(p_cx0: np.ndarray) -> None:
    """Validate the initial cumulative damage probability vector.

    Cx0 must be deterministic: exactly one entry equal to 1.0, all
    others zero, length 6, no negatives.
    """
    if len(p_cx0) != 6:
        raise ValueError(
            f"Cx0 probability vector must have exactly 6 elements, got {len(p_cx0)}."
        )
    if np.any(p_cx0 < 0):
        raise ValueError("Cx0 probability vector must not contain negative values.")

    non_zero_count = np.count_nonzero(p_cx0)
    if non_zero_count == 0:
        raise ValueError(
            "Cx0 probability vector must have exactly one entry of 1.0, "
            "but all entries are zero."
        )
    if non_zero_count > 1:
        raise ValueError(
            f"Cx0 probability vector must have exactly one non-zero entry, "
            f"but found {non_zero_count} non-zero entries: {p_cx0}."
        )

    non_zero_val = p_cx0[p_cx0 != 0][0]
    if not np.isclose(non_zero_val, 1.0):
        raise ValueError(
            f"The single non-zero entry in Cx0 must equal 1.0, got {non_zero_val}."
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_temporal_bn(
    n_components: int,
    p_cx0: np.ndarray = None,
    p_wx: np.ndarray = None,
    p_lx: np.ndarray = None,
    p_px: np.ndarray = None,
    p_vx: np.ndarray = None,
    p_ux: np.ndarray = None,
) -> tuple[dict, dict]:
    """Build a temporal Bayesian Network with shared Ux and per-component Vx, Zx.

    Args:
        n_components: Number of temporal instances (>= 1).
        p_cx0: Length-6 deterministic vector for initial cumulative damage.
        p_wx, p_lx: Length-6 probability vectors for weather and loading.
        p_px: Length-2 probability vector for previous-repair indicator.
        p_vx, p_ux: Length-5 marginal probability vectors for noise variables.

    Returns:
        Tuple (varis, cpms) of variable and CPM dictionaries.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}.")

    if p_cx0 is None:
        p_cx0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if p_wx is None:
        p_wx = DEFAULT_WX_PROBS.copy()
    if p_lx is None:
        p_lx = DEFAULT_LX_PROBS.copy()
    if p_px is None:
        p_px = DEFAULT_PX_PROBS.copy()
    if p_vx is None:
        p_vx = DEFAULT_VX_PROBS.copy()
    if p_ux is None:
        p_ux = DEFAULT_UX_PROBS.copy()

    validate_cx0_probs(p_cx0)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    varis = {}

    varis['Cx0'] = variable.Variable('Cx0', DAMAGE_STATES)
    varis['Ux'] = variable.Variable('Ux', NOISE_STATES)

    for i in range(1, n_components + 1):
        varis[f'Wx{i}']  = variable.Variable(f'Wx{i}',  DAMAGE_STATES)
        varis[f'Lx{i}']  = variable.Variable(f'Lx{i}',  DAMAGE_STATES)
        varis[f'Tx{i}']  = variable.Variable(f'Tx{i}',  DAMAGE_STATES)
        varis[f'Px{i}']  = variable.Variable(f'Px{i}',  REPAIR_STATES)
        varis[f'Cx{i}']  = variable.Variable(f'Cx{i}',  DAMAGE_STATES)
        varis[f'Clx{i}'] = variable.Variable(f'CLx{i}', DAMAGE_STATES)
        varis[f'Rx{i}']  = variable.Variable(f'Rx{i}',  RESISTANCE_STATES)
        varis[f'Zx{i}']  = variable.Variable(f'Zx{i}',  NOISE_STATES)
        varis[f'Vx{i}']  = variable.Variable(f'Vx{i}',  NOISE_STATES)

    # ------------------------------------------------------------------
    # CPMs
    # ------------------------------------------------------------------
    cpms = {}
    six_states = np.array([[0],[1],[2],[3],[4],[5]], dtype=int)
    five_states = np.array([[0],[1],[2],[3],[4]], dtype=int)

    cpms['Cx0'] = cpm.Cpm(
        variables=[varis['Cx0']],
        no_child=1,
        C=six_states,
        p=p_cx0,
    )

    cpms['Ux'] = cpm.Cpm(
        variables=[varis['Ux']],
        no_child=1,
        C=five_states,
        p=p_ux.copy(),
    )

    for i in range(1, n_components + 1):
        cpms[f'Wx{i}'] = cpm.Cpm(
            variables=[varis[f'Wx{i}']],
            no_child=1, C=six_states, p=p_wx.copy(),
        )
        cpms[f'Lx{i}'] = cpm.Cpm(
            variables=[varis[f'Lx{i}']],
            no_child=1, C=six_states, p=p_lx.copy(),
        )
        cpms[f'Tx{i}'] = cpm.Cpm(
            variables=[varis[f'Tx{i}'], varis[f'Wx{i}'], varis[f'Lx{i}']],
            no_child=1, C=TX_C_MATRIX.copy(),
            p=np.ones(TX_C_MATRIX.shape[0]),
        )
        cpms[f'Px{i}'] = cpm.Cpm(
            variables=[varis[f'Px{i}']],
            no_child=1,
            C=np.array([[0],[1]], dtype=int),
            p=p_px.copy(),
        )
        cpms[f'Cx{i}'] = cpm.Cpm(
            variables=[varis[f'Cx{i}'], varis[f'Px{i}'],
                       varis[f'Cx{i-1}'], varis[f'Tx{i}']],
            no_child=1, C=CX_C_MATRIX.copy(),
            p=np.ones(CX_C_MATRIX.shape[0]),
        )
        cpms[f'Vx{i}'] = cpm.Cpm(
            variables=[varis[f'Vx{i}']],
            no_child=1, C=five_states, p=p_vx.copy(),
        )
        cpms[f'Zx{i}'] = cpm.Cpm(
            variables=[varis[f'Zx{i}'], varis[f'Vx{i}'], varis['Ux']],
            no_child=1, C=ZX_C_MATRIX.copy(),
            p=np.ones(ZX_C_MATRIX.shape[0]),
        )
        cpms[f'Rx{i}'] = cpm.Cpm(
            variables=[varis[f'Rx{i}'], varis[f'Zx{i}']],
            no_child=1, C=RX_C_MATRIX.copy(),
            p=np.ones(RX_C_MATRIX.shape[0]),
        )
        cpms[f'Clx{i}'] = cpm.Cpm(
            variables=[varis[f'Clx{i}'], varis[f'Cx{i}'], varis[f'Rx{i}']],
            no_child=1, C=CLX_C_MATRIX.copy(),
            p=np.ones(CLX_C_MATRIX.shape[0]),
        )

    return varis, cpms


# ---------------------------------------------------------------------------
# Elimination order
# ---------------------------------------------------------------------------

def build_elimination_order(varis: dict, n_components: int) -> list:
    """Build the variable elimination order matching the original script."""
    causes = ['Wx', 'Lx', 'Tx']
    interventions = ['Px']
    unique_rv = ['Vx']
    auxiliary = ['Zx']
    resistance = ['Rx']
    observations = ['Clx']
    effects = ['Cx']

    order = []
    order += [varis[f'{c}{i}'] for c in causes for i in range(1, n_components + 1)]
    order += [varis[f'{k}{i}'] for k in interventions for i in range(1, n_components + 1)]
    order += [varis['Ux']]
    order += [varis[f'{v}{i}'] for v in unique_rv for i in range(1, n_components + 1)]
    order += [varis[f'{z}{i}'] for z in auxiliary for i in range(1, n_components + 1)]
    order += [varis[f'{r}{i}'] for r in resistance for i in range(1, n_components + 1)]
    order += [varis[f'{o}{i}'] for o in observations for i in range(1, n_components)]
    order += [varis[f'{e}{i}'] for e in effects for i in range(0, n_components + 1)]
    return order
