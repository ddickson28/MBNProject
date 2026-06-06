import torch
from torch.distributions import Normal
from torch.distributions import HalfNormal

# One-sided truncated Normal on [0, +inf). PyTorch has no built-in TruncatedNormal
# distribution object, so this is implemented via the parent Normal's cdf/icdf.

class TruncNormalPos:
    def __init__(self, loc, scale, validate_args=False):
        self.base = Normal(loc, scale, validate_args=validate_args)
        zero = torch.zeros_like(loc)
        self.Phi_a = self.base.cdf(zero)
        self.Z = (1.0 - self.Phi_a).clamp_min(1e-12)
        self.log_Z = torch.log(self.Z)

    def sample(self):
        u = torch.rand_like(self.base.loc)
        q = (self.Phi_a + u * self.Z).clamp(min=1e-12, max=1.0 - 1e-12)
        return self.base.icdf(q)

    def log_prob(self, x):
        lp = self.base.log_prob(x) - self.log_Z
        return torch.where(x >= 0, lp, torch.full_like(lp, -float("inf")))

# Define custom variables and probability distributions

#Continuous - Weather Loading
class Wx:
    def __init__(self, childs, parents=[], sigma=1.0, device="cpu"):
        """
        Wx ~ HalfNormal(0, Scale) Scale = Sigma^2

        childs  : [Variable Wx]
        parents : [empty]
        mean    : fixed mean (float)
        sigma   : fixed noise std (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

    #    self.mean = float(mean)
        self.sigma = float(sigma)

    # ------------------------------------------------------------------

    def sample(self, n_sample):
        """
        N : int
            number of samples to draw

        Returns
        -------
        Cs   : (N,) sampled Wx values
        logp : (N,) log p(Wx)
        """

    #    mean = torch.full((n_sample,), self.mean, device=self.device)
        std = torch.full((n_sample,), self.sigma, device=self.device)    
    #    std = torch.full_like(mean, self.sigma, device=self.device)

        dist = HalfNormal(std)
        Cs = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (n_sample, )

        Returns
        -------
        log p(Wx) : (n_sample,)
        """
        Cs = Cs.to(self.device)

        Wx_val = Cs

        #    mean = torch.full_like(Cs, self.mean) 
        std = torch.full_like(Cs, self.sigma)

        dist = HalfNormal(std)
        return dist.log_prob(Wx_val)

#Continuous - Cargo Loading
class Lx:
    def __init__(self, childs, parents=[], sigma=1.0, device="cpu"):
        """
        Lx ~ HalfNormal(scale=sigma)    # equivalent to |N(0, sigma^2)|

        childs  : [Variable Lx]
        parents : [empty]
        sigma   : scale of the underlying Normal (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.sigma = float(sigma)


    # ------------------------------------------------------------------
    def sample(self, n_sample):
        """
        n_sample : int
            number of samples to draw

        Returns
        -------
        Cs   : (n_sample,) sampled Lx values
        logp : (n_sample,) log p(Lx)
        """
        std = torch.full((n_sample,), self.sigma, device=self.device)

        dist = HalfNormal(std)

        Cs = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (n_sample, )

        Returns
        -------
        log p(Lx) : (n_sample,)
        """
        Cs = Cs.to(self.device)

        std = torch.full_like(Cs, self.sigma)

        dist = HalfNormal(std, validate_args=False)
        logp = dist.log_prob(Cs)
        return torch.where(Cs >= 0, logp, torch.full_like(logp, -float("inf")))

#Continuous - Total Loading | Weather, Cargo
class Tx:
    def __init__(self, childs, parents, device="cpu"):
        """
        Tx = Wx + Lx  (deterministic; no added noise)

        All randomness in Tx comes from its parents Wx and Lx.

        childs  : [Variable Tx]
        parents : [Variable Wx, Variable Lx]
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.Wx = parents[0]
        self.Lx = parents[1]

    # ------------------------------------------------------------------

    def sample(self, Cs_pars):
        """
        Cs_pars : (n_sample, 2)
            Cs_pars[:,0] = Wx value
            Cs_pars[:,1] = Lx value

        Returns
        -------
        Cs   : Wx + Lx = Tx values
        logp : log p(Tx | Wx,Lx)
        """
        Cs_pars = Cs_pars.to(self.device)

        Wx_val = Cs_pars[:, 0]
        Lx_val = Cs_pars[:, 1]

        Cs = Wx_val + Lx_val
        logp = torch.zeros_like(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (n_sample, 3)
            Cs[:,0] = Tx value
            Cs[:,1] = Wx value
            Cs[:,2] = Lx value

        Returns
        -------
        log p(Tx | Wx,Lx) : (n_sample,)
            0    where Tx ≈ Wx + Lx (constraint satisfied)
            -inf where Tx ≠ Wx + Lx (constraint violated)
        """
        Cs = Cs.to(self.device)

        C_val  = Cs[:, 0]
        Wx_val = Cs[:, 1]
        Lx_val = Cs[:, 2]

        on_surface = torch.isclose(C_val, Wx_val + Lx_val)
        return torch.where(on_surface,
                           torch.zeros_like(C_val),
                           torch.full_like(C_val, -float("inf")))

#Continuous - Cumulative damage at time 0
class Cx0:
    def __init__(self, childs, parents=[], sigma=1.0, device="cpu"):
        """
        Cx0 ~ HalfNormal(scale=sigma)    # equivalent to |N(0, sigma^2)|

        childs  : [Variable Cx0]
        parents : [empty]
        sigma   : scale of the underlying Normal (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.sigma = float(sigma)

    # ------------------------------------------------------------------

    def sample(self, n_sample):
        """
        n_sample : int
            number of samples to draw

        Returns
        -------
        Cs   : (n_sample,) sampled Cx0 values
        logp : (n_sample,) log p(Cx0)
        """

        std = torch.full((n_sample,), self.sigma, device=self.device)

        dist = HalfNormal(std)
        Cs = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (n_sample, )

        Returns
        -------
        log p(Cx0) : (n_sample,)
        """
        Cs = Cs.to(self.device)

        std = torch.full_like(Cs, self.sigma)

        dist = HalfNormal(std, validate_args=False)
        logp = dist.log_prob(Cs)
        return torch.where(Cs >= 0, logp, torch.full_like(logp, -float("inf")))

#Continuous - Cumulative damage | Total loading, Previous Repair
class Cx:
    def __init__(self, childs, parents, device='cpu'):
        """

        childs: list [Cx]
        parents: list [Tx, Px]
            Tx: continuous-valued parent (tensor-like values for samples)
            Px: binary parent (0 or 1)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

    # ------------------------------------------------------------------
    def sample(self, Cs_pars):
        """
        Cs_pars: (N, 2)
            Cs_pars[:,0] = Tx value   (float)
            Cs_pars[:,1] = Px index   (0 or 1)
        
        Returns:
            Cx samples (N,)
        """
        Cs_pars = Cs_pars.to(self.device)

        Tx_val = Cs_pars[:, 0]
        Px_idx = Cs_pars[:, 1].long()

        # Cx = Tx if Px==0, else 0
        Cs = torch.where(Px_idx == 0, Tx_val, torch.zeros_like(Tx_val))

        # deterministic function, i.e. P(Cx | Tx, Px) = 1
        n_sample = Cs_pars.shape[0]
        ps = torch.log(torch.ones(n_sample,)).to(self.device)

        return Cs, ps

    # ------------------------------------------------------------------
    def log_prob(self, Cxs):
        """
        Cxs: shape (N, 3)
            Cxs[:,0] = Cx value
            Cxs[:,1] = Tx value 
            Cxs[:,2] = Px state (0 or 1) 
        
        Returns:
            log p(Cx | Tx, Px) of shape (N,)
        """

        Cxs = Cxs.to(self.device)

        Cx_val = Cxs[:, 0]
        Tx_val = Cxs[:, 1]
        Px_idx = Cxs[:, 2].long()

        # Deterministic rule: valid_Cx = Tx if Px==0 else 0
        expected_Cx = torch.where(Px_idx == 0, Tx_val, torch.zeros_like(Tx_val))

        # Valid if Cx_val == expected_Cx
        is_valid = torch.isclose(Cx_val, expected_Cx, rtol=1e-5, atol=1e-8) # Add some acceptable error margin for floating point comparisons if needed.

        # log 1 = 0 for valid, log 0 = -inf for invalid
        logp = torch.where(is_valid, torch.zeros_like(Cx_val), torch.full_like(Cx_val, -float("inf")))

        return logp

#Continuous - Crack location  | Cumulative damage, Resistance
class Clx:
    def __init__(self, childs, parents, device='cpu'):
        """

        childs: list [Clx]
        parents: list [Cx, Rx]
            Cx: continuous-valued parent (tensor-like values for samples)
            Rx: continuous-valued parent (tensor-like values for samples)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

    # ------------------------------------------------------------------
    def sample(self, Cs_pars):
        """
        Cs_pars: (N, 2)
            Cs_pars[:,0] = Cx value   (float)
            Cs_pars[:,1] = Rx value   (float)

        Returns:
            Clx samples (N,)
        """
        Cs_pars = Cs_pars.to(self.device)

        Cx_val = Cs_pars[:, 0]
        Rx_val = Cs_pars[:, 1]

        # Clx = 1 if Cx > Rx else 0
        Cs = torch.where(Cx_val > Rx_val, torch.ones_like(Cx_val), torch.zeros_like(Cx_val))


        # deterministic function, i.e. P(Clx | Cx, Rx) = 1
        n_sample = Cs_pars.shape[0]
        ps = torch.log(torch.ones(n_sample,)).to(self.device)

        return Cs, ps

    # ------------------------------------------------------------------
    def log_prob(self, Clxs):
        """
        Clxs: shape (N, 3)
            Clxs[:,0] = Clx value
            Clxs[:,1] = Cx value 
            Clxs[:,2] = Rx value 
        
        Returns:
            log p(Clx | Cx, Rx) of shape (N,)
        """

        Clxs = Clxs.to(self.device)

        Clx_val = Clxs[:, 0]
        Cx_val = Clxs[:, 1]
        Rx_val = Clxs[:, 2]

        # Clx = 1 if Cx > Rx else 0
        # Deterministic rule: valid_Clx = 1 if Cx > Rx else 0
        expected_Clx = torch.where(Cx_val > Rx_val, torch.ones_like(Cx_val), torch.zeros_like(Cx_val))

        # Valid if Clx_val == expected_Clx
        is_valid = torch.isclose(Clx_val, expected_Clx, rtol=1e-5, atol=1e-8) # Corrected Add some acceptable error margin for floating point comparisons if needed.

        # log 1 = 0 for valid, log 0 = -inf for invalid
        logp = torch.where(is_valid, torch.zeros_like(Clx_val), torch.full_like(Clx_val, -float("inf")))

        return logp

#Continuous - Resistance Rx|Z Rx = 1 + Zx
class Rx:
    def __init__(self, childs, parents, device="cpu"):
        """
        Rx = exp(Zx)  (deterministic; log-normal w.r.t. a Normal Zx)

        Always positive regardless of the sign of Zx.

        childs  : [Variable Rx]
        parents : [Variable Zx]
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        # parent variables
        self.Zx = parents[0]

    # ------------------------------------------------------------------

    def sample(self, Cs_pars):
        """
        Cs_pars : (N, 1)
            Cs_pars[:,0] = Zx value

        Returns
        -------
        Cs   : (N,) Rx values (= exp(Zx))
        logp : (N,) log p(Rx | Zx) — zero, deterministic node
        """
        Cs_pars = Cs_pars.to(self.device)

        Zx_val = Cs_pars[:, 0]

        Cs   = torch.exp(Zx_val)
        logp = torch.zeros_like(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (N, 2)
            Cs[:,0] = Rx value
            Cs[:,1] = Zx value

        Returns
        -------
        log p(Rx | Zx) : (N,)
            0    where Rx ≈ exp(Zx) (constraint satisfied)
            -inf otherwise
        """
        Cs = Cs.to(self.device)

        C_val  = Cs[:, 0]
        Zx_val = Cs[:, 1]

        expected = torch.exp(Zx_val)
        on_surface = torch.isclose(C_val, expected)
        return torch.where(on_surface,
                           torch.zeros_like(C_val),
                           torch.full_like(C_val, -float("inf")))

#Continuous - Auxiliary variable Zx | Ux, Vx
class Zx:
    def __init__(self, childs, parents, rho=0.2, device="cpu"):
        """
        Zx = sqrt(1 - rho**2) * Vx + rho * Ux  (deterministic; no added noise)

        All randomness in Zx comes from its parents Ux and Vx.

        childs  : [Variable Zx]
        parents : [Variable Ux, Variable Vx]
        rho     : correlation coefficient (float), default 0.2
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.rho = float(rho)
        self._a  = (1.0 - self.rho ** 2) ** 0.5   # coefficient on Vx

        # parent variables
        self.Ux = parents[0]
        self.Vx = parents[1]

    # ------------------------------------------------------------------

    def sample(self, Cs_pars):
        """
        Cs_pars : (n_sample, 2)
            Cs_pars[:,0] = Ux value
            Cs_pars[:,1] = Vx value

        Returns
        -------
        Cs   : (n_sample,) Zx values (= sqrt(1-rho**2)*Vx + rho*Ux)
        logp : (n_sample,) log p(Zx | Ux,Vx) — zero, deterministic node
        """
        Cs_pars = Cs_pars.to(self.device)

        Ux_val = Cs_pars[:, 0]
        Vx_val = Cs_pars[:, 1]

        Cs   = self._a * Vx_val + self.rho * Ux_val
        logp = torch.zeros_like(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (n_sample, 3)
            Cs[:,0] = Zx value
            Cs[:,1] = Ux value
            Cs[:,2] = Vx value

        Returns
        -------
        log p(Zx | Ux,Vx) : (n_sample,)
            0    where Zx ≈ sqrt(1-rho**2)*Vx + rho*Ux (constraint satisfied)
            -inf otherwise
        """
        Cs = Cs.to(self.device)

        C_val  = Cs[:, 0]
        Ux_val = Cs[:, 1]
        Vx_val = Cs[:, 2]

        expected = self._a * Vx_val + self.rho * Ux_val
        on_surface = torch.isclose(C_val, expected)
        return torch.where(on_surface,
                           torch.zeros_like(C_val),
                           torch.full_like(C_val, -float("inf")))

#Continuous - Latent unobservable gaussian correlates across locations
class Ux:
    def __init__(self, childs, parents=[], sigma=0.2, device="cpu"):
        """
        Ux ~ Normal(0, sigma^2)

        childs  : [Variable Ux]
        parents : [empty]
        sigma   : noise std (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.sigma = float(sigma)


    # ------------------------------------------------------------------
    def sample(self, n_sample):
        """
        n_sample : int
            number of samples to draw

        Returns
        -------
        Cs   : (n_sample,) sampled Ux values
        logp : (n_sample,) log p(Ux)
        """
        mean = torch.zeros(n_sample, device=self.device)
        std  = torch.full((n_sample,), self.sigma, device=self.device)

        dist = Normal(mean, std)

        Cs = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (n_sample, )

        Returns
        -------
        log p(Ux) : (n_sample,)
        """
        Cs = Cs.to(self.device)

        mean = torch.zeros_like(Cs)
        std  = torch.full_like(Cs, self.sigma)

        dist = Normal(mean, std)
        return dist.log_prob(Cs)

#Continuous - Latent unobservable gaussian unique to each location
class Vx:
    def __init__(self, childs, parents=[], sigma=0.2, device="cpu"):
        """
        Vx ~ Normal(0, sigma^2)

        childs  : [Variable Vx]
        parents : [empty]
        sigma   : noise std (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.sigma = float(sigma)


    # ------------------------------------------------------------------
    def sample(self, n_sample):
        """
        n_sample : int
            number of samples to draw

        Returns
        -------
        Cs   : (n_sample,) sampled Vx values
        logp : (n_sample,) log p(Vx)
        """
        mean = torch.zeros(n_sample, device=self.device)
        std  = torch.full((n_sample,), self.sigma, device=self.device)

        dist = Normal(mean, std)

        Cs = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (n_sample, )

        Returns
        -------
        log p(Vx) : (n_sample,)
        """
        Cs = Cs.to(self.device)

        mean = torch.zeros_like(Cs)
        std  = torch.full_like(Cs, self.sigma)

        dist = Normal(mean, std)
        return dist.log_prob(Cs)


# Defining variables and probabilities
import os, sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE)

repo_root = os.path.abspath(os.path.join(BASE, "../.."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from tbnpy import cpt, variable
import numpy as np
import torch

device = ('cuda' if os.environ.get('USE_CUDA', '0') == '1' else 'cpu')

def define_variables():
    varis = {}

    varis['Wx'] = variable.Variable(name='Wx', values=(0, torch.inf))  # Continuous Half Normal

    varis['Lx'] = variable.Variable(name='Lx', values=(0, torch.inf))  # Continuous Half Normal

    varis['Tx'] = variable.Variable(name='Tx', values=(0, torch.inf))  # Continuous Half Normal

    varis['Cx'] = variable.Variable(name='Cx', values=(0, torch.inf))  # Continuous Half Normal

    varis['Px'] = variable.Variable(name='Px', values=['False', 'True']) # Boolean 

    varis['Cx0'] = variable.Variable(name='Cx0', values=(0, torch.inf))  # Continuous Half Normal

    varis['Clx'] = variable.Variable(name='Clx', values=['False', 'True'])  # Boolean 

    varis['Rx'] = variable.Variable(name='Rx', values=(0, torch.inf))  # Continuous Log Normal

    varis['Zx'] = variable.Variable(name='Zx', values=(-torch.inf, torch.inf))  # Continuous Normal

    varis['Ux'] = variable.Variable(name='Ux', values=(-torch.inf, torch.inf))  # Continuous Normal

    varis['Vx'] = variable.Variable(name='Vx', values=(-torch.inf, torch.inf))  # Continuous Normal

    return varis

def define_probs(varis, device='cpu'):
    probs = {}

    probs['Wx'] = Wx(childs=[varis['Wx']], device=device)

    probs['Lx'] = Lx(childs=[varis['Lx']], device=device)

    probs['Tx'] = Tx(childs=[varis['Tx']], parents=[varis['Wx'], varis['Lx']] , device=device)

    probs['Cx'] = Cx(childs=[varis['Cx']], parents=[varis['Tx'], varis['Px']], device=device)

    probs['Px'] = cpt.Cpt(childs=[varis['Px']], C=np.array([[0], [1]]), p=np.array([1.0, 0.0]), device=device) 

    probs['Clx'] = Clx(childs=[varis['Clx']], parents=[varis['Cx'], varis['Rx']], device=device) 

    probs['Cx0'] = Cx0(childs=[varis['Cx0']], device=device)

    probs['Rx'] = Rx(childs=[varis['Rx']], parents=[varis['Zx']], device=device)

    probs['Zx'] = Zx(childs=[varis['Zx']], parents=[varis['Ux'], varis['Vx']], device=device)

    probs['Ux'] = Ux(childs=[varis['Ux']], device=device)

    probs['Vx'] = Vx(childs=[varis['Vx']], device=device)
    
    return probs

# Inference

import os, sys
from pathlib import Path
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True, parents=True) #Added
sys.path.append(BASE)

repo_root = os.path.abspath(os.path.join(BASE, "../.."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tbnpy import inference, adaptiveMH
#from s1_define_model import define_variables, define_probs

"""
Overall structure:
s2_run_sample.py
├─ define variables & probs   (already done)
├─ define evidence
├─ forward sampling (initialisation)
├─ adaptive MH run
├─ posterior extraction
└─ plotting
"""

def define_evidence(n_evi=10, seed=123):
    """
    Evidence DataFrame: shape (n_evi, n_evidence_vars)

    OC ~ Normal(0, 0.15)
    """
    rng = np.random.default_rng(seed)

    evidence = pd.DataFrame({
        "OC": rng.normal(loc=-0.3, scale=0.05, size=n_evi)
    })

    return evidence

def sample_prior(probs, variables, n_sample=5000):
    """
    Sample prior for all variables using forward sampling.

    Parameters
    ----------
    probs : dict
        BN probability objects
    variables : dict or list
        Variable objects
    n_sample : int
        Number of prior samples

    Returns
    -------
    dict[var_name -> np.ndarray]
        Each array has shape (n_sample,)
    """
    if isinstance(variables, dict):
        var_list = list(variables.values())
    else:
        var_list = list(variables)

    query_nodes = [v.name for v in var_list]

    filled = inference.sample(
        probs=probs,
        query_nodes=query_nodes,
        n_sample=n_sample,
    )

    prior = {}
    for prob in filled.values():
        Cs = prob.Cs
        for j, child_var in enumerate(prob.childs):
            name = child_var.name
            if Cs.ndim == 1:
                prior[name] = Cs.detach().cpu().numpy()
            else:                       # 2-D (n_sample, dim)
                prior[name] = Cs[:, j].detach().cpu().numpy()

    return prior

def forward_initialise(probs, latent_vars, evidence, n_chain):
    """
    Use forward sampling to initialise MCMC chains.
    """
    probs_copy = inference.sample_evidence(
        probs=probs,
        query_nodes=[v.name for v in latent_vars],
        n_sample=n_chain,
        evidence_df=evidence,
    )
    return probs_copy

def run_mcmc(probs, varis, evidence, update_blocks, burnin=200, n_chain=5000, n_iter=2000, progress_every=100):
    sampler = adaptiveMH.HybridAdaptiveMH(
        probs=probs,
        variables=list(varis.values()),
        evidence_df=evidence,
        n_chain=n_chain,
        adapt=adaptiveMH.AdaptConfig(
            burnin=burnin,
            gamma=0.6,
            target_accept=0.234,
            alpha=0.5,
        ),
    )

    # --- Initialise from forward samples ---
    probs_copy = forward_initialise(
        probs,
        sampler.latent_vars,
        evidence,
        n_chain,
    )
    sampler.init_state_from_forward_samples(probs_copy)

    # --- Run MCMC ---
    out = sampler.run(
        n_iter=n_iter,
        store_every=10,   # thin
        update_blocks=update_blocks,
        progress_every=progress_every,
    )

    return sampler, out

def extract_posterior(sampler):
    """
    Returns dict[var_name -> 1D np.ndarray]
    (all evidence rows and chains flattened)
    """
    posterior = {}

    for v in sampler.latent_vars:
        x = sampler.state[v.name]  # (n_evi, n_chain)
        posterior[v.name] = x.detach().cpu().numpy().reshape(-1)

    return posterior

import numpy as np
import matplotlib.pyplot as plt

def plot_prior_vs_posterior(prior, posterior, var, bins=60, fname: str = None):
    """
    Plot prior vs posterior for one variable.

    Parameters
    ----------
    prior : dict[str, np.ndarray]
        Prior samples for all variables
    posterior : dict[str, np.ndarray]
        Posterior samples for all variables
    var : Variable
        tbnpy Variable object
    bins : int
        Number of bins for continuous histograms
    fname : str, optional
        File name to save the plot (saved in RESULTS folder). If None, the plot is not saved.
    """
    name = var.name

    if name not in prior:
        raise KeyError(f"Variable '{name}' not found in prior samples.")
    if name not in posterior:
        raise KeyError(f"Variable '{name}' not found in posterior samples.")

    x_prior = prior[name]
    x_post = posterior[name]

    plt.figure(figsize=(5, 4))

    # Discrete variable
    if isinstance(var.values, list):
        K = len(var.values)
        bins_disc = np.arange(K + 1) - 0.5

        plt.hist(
            x_prior,
            bins=bins_disc,
            density=True,
            alpha=0.5,
            label="Prior",
            color="gray",
        )

        plt.hist(
            x_post,
            bins=bins_disc,
            density=True,
            alpha=0.6,
            label="Posterior",
            color="tab:blue",
        )

        plt.xticks(range(K), var.values)
        plt.ylabel("Probability")

    # Continuous variable
    else:
        plt.hist(
            x_prior,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Prior",
            color="gray",
        )

        plt.hist(
            x_post,
            bins=bins,
            density=True,
            alpha=0.6,
            label="Posterior",
            color="tab:blue",
        )

        plt.ylabel("Density")

    plt.xlabel(name)
    plt.legend()
    plt.tight_layout()
    if fname is not None:
        plt.savefig(RESULTS / fname, dpi=300)

if __name__ == "__main__":
    device = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

    varis = define_variables()
    probs = define_probs(varis, device=device)

    # Prior distribution without evidence
    prior = sample_prior(probs, varis, n_sample=10_000)

    # --- TEMP: forward inference only, skip MCMC below -------------------
    print("\nPrior summary (n=10,000 forward samples):")
    for name, samples in prior.items():
        print(f"  {name:>4}: mean={samples.mean():+.3f}  "
              f"std={samples.std():.3f}  "
              f"min={samples.min():+.3f}  max={samples.max():+.3f}")

    # Prior-only histograms (one PNG per variable in RESULTS folder)
    for name, samples in prior.items():
        var = varis[name]
        plt.figure(figsize=(5, 4))
        if isinstance(var.values, list):
            K = len(var.values)
            bins_disc = np.arange(K + 1) - 0.5
            plt.hist(samples, bins=bins_disc, density=True, color="gray", alpha=0.7)
            plt.xticks(range(K), var.values)
            plt.ylabel("Probability")
        else:
            plt.hist(samples, bins=60, density=True, color="gray", alpha=0.7)
            plt.ylabel("Density")
        plt.xlabel(name)
        plt.title(f"Prior — {name}")
        plt.tight_layout()
        plt.savefig(RESULTS / f"prior_{name}.png", dpi=300)
        plt.close()
    print(f"\nSaved {len(prior)} prior plots to: {RESULTS}")

    import sys; sys.exit(0)
    # --- end TEMP --------------------------------------------------------

    # Posterior inference with evidence
    n_evi = 20
    evidence = define_evidence(n_evi=n_evi)

    n_chain = 100
    n_iter = 30 # 30_000
    burnin = 200

    query_varis = {v: varis[v] for v in ['Wx', 'Lx', 'Tx', 'Cx', 'Px','Clx', 'Cx0', 'Rx', 'Zx', 'Ux', 'Vx']} # 
    query_probs = {k: v for k, v in probs.items() if any(c.name in query_varis for c in v.childs)}
    update_blocks = ['Wx', 'Lx', 'Tx', 'Cx', 'Px','Clx', 'Cx0', 'Rx', 'Zx', 'Ux', 'Vx'] 

    sampler, out = run_mcmc(
        query_probs,
        query_varis,
        evidence,
        update_blocks = update_blocks,
        burnin=burnin,
        n_chain=n_chain,
        n_iter=n_iter,
        progress_every = 100
    )

    posterior = extract_posterior(sampler)
    for _, v in query_varis.items():
        if v.name in evidence.columns:
            continue  # skip evidence variables
        plot_prior_vs_posterior(prior, posterior, v, fname=r"plot_" + v.name + f"_{n_evi}_evi_{burnin}_burnin_{n_chain}_chains_{n_iter}_iters" + ".png")

#    print("Acceptance rates:")
#    for k, v in out["accept_rate"].items():
        print(f"  {k}: {v:.3f}")

    query_varis = {v: varis[v] for v in ['Wx', 'Lx', 'Tx', 'Cx', 'Px','Clx', 'Cx0', 'Rx', 'Zx', 'Ux', 'Vx']} 
    query_probs = {k: v for k, v in probs.items() if any(c.name in query_varis for c in v.childs)}
    update_blocks = ['Wx', 'Lx', 'Tx', 'Cx', 'Px','Clx', 'Cx0', 'Rx', 'Zx', 'Ux', 'Vx'] 