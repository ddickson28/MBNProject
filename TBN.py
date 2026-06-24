import torch
from torch.distributions import Normal
from torch.distributions import LogNormal
from torch.distributions import Beta

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
    def __init__(self, childs, parents=[], mean=0.0, sigma=1.0, device="cpu"):
        """
        Wx ~ Log-Normal(0, 1) # equivalent to exp(N(0, 1))

        childs  : [Variable Wx]
        parents : [empty]
        mean    : fixed mean (float)
        sigma   : fixed noise std (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.mean = float(mean)
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


        std = torch.full((n_sample,), self.sigma, device=self.device)
        mean = torch.full_like(std, self.mean)
        dist = LogNormal(mean, std)
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
        mean = torch.full_like(Cs, self.mean)

        dist = LogNormal(mean, std)
        return dist.log_prob(Wx_val)

#Continuous - Cargo Loading
class Lx:
    def __init__(self, childs, parents=[], mean=0.0, sigma=1.0, device="cpu"):
        """
        Lx ~ Log-Normal(mean, sigma) # equivalent to exp(N(mean, sigma^2))

        childs  : [Variable Lx]
        parents : [empty]
        mean    : mean of the underlying Normal (float)
        sigma   : std of the underlying Normal (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.mean  = float(mean)
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
        std  = torch.full((n_sample,), self.sigma, device=self.device)
        mean = torch.full_like(std, self.mean)

        dist = LogNormal(mean, std)
        Cs   = dist.sample()
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

        std  = torch.full_like(Cs, self.sigma)
        mean = torch.full_like(Cs, self.mean)

        dist = LogNormal(mean, std)
        return dist.log_prob(Cs)

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

#Continuous - Cumulative damage at time 0 (root, log-normal)
class Cx0:
    def __init__(self, childs, parents=[], mean=0.0, sigma=1.0, device="cpu"):
        """
        Cx0 ~ LogNormal(mean, sigma)  # equivalent to exp(N(mean, sigma^2))

        childs  : [Variable Cx0]
        parents : [empty]
        mean    : mean of the underlying Normal (float)
        sigma   : std of the underlying Normal (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.mean  = float(mean)
        self.sigma = float(sigma)

    # ------------------------------------------------------------------
    def sample(self, n_sample):
        """
        n_sample : int

        Returns
        -------
        Cs   : (n_sample,) sampled Cx0 values
        logp : (n_sample,) log p(Cx0)
        """
        std  = torch.full((n_sample,), self.sigma, device=self.device)
        mean = torch.full_like(std, self.mean)

        dist = LogNormal(mean, std)
        Cs   = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (n_sample,)

        Returns
        -------
        log p(Cx0) : (n_sample,)
        """
        Cs = Cs.to(self.device)

        std  = torch.full_like(Cs, self.sigma)
        mean = torch.full_like(Cs, self.mean)

        dist = LogNormal(mean, std)
        return dist.log_prob(Cs)

#Continuous - Cumulative damage | Previous damage, Total loading, Repair flag
class Cx:
    def __init__(self, childs, parents, device='cpu'):
        """
        Cx{t} = Cx{t-1} + Tx  if Px == 0 (no repair)
        Cx{t} = 0              if Px == 1 (repair resets damage)

        childs:  list [Cx]
        parents: list [Cx_prev, Tx, Px]
            Cx_prev: continuous previous-timestep damage
            Tx:      continuous total loading
            Px:      binary repair flag (0 = no repair, 1 = repair)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

    # ------------------------------------------------------------------
    def sample(self, Cs_pars):
        """
        Cs_pars: (N, 3)
            Cs_pars[:,0] = Cx_prev value  (float)
            Cs_pars[:,1] = Tx value       (float)
            Cs_pars[:,2] = Px index       (0 or 1)

        Returns:
            Cs   : (N,) Cx samples
            logp : (N,) zeros  (deterministic node)
        """
        Cs_pars = Cs_pars.to(self.device)

        Cx_prev = Cs_pars[:, 0]
        Tx_val  = Cs_pars[:, 1]
        Px_idx  = Cs_pars[:, 2].long()

        Cs = torch.where(Px_idx == 0, Cx_prev + Tx_val, torch.zeros_like(Tx_val))
        logp = torch.zeros(Cs_pars.shape[0], device=self.device)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cxs):
        """
        Cxs: shape (N, 4)
            Cxs[:,0] = Cx value      (child)
            Cxs[:,1] = Cx_prev value (parent 0)
            Cxs[:,2] = Tx value      (parent 1)
            Cxs[:,3] = Px state      (parent 2: 0 or 1)

        Returns:
            log p(Cx | Cx_prev, Tx, Px) of shape (N,)
        """
        Cxs = Cxs.to(self.device)

        Cx_val  = Cxs[:, 0]
        Cx_prev = Cxs[:, 1]
        Tx_val  = Cxs[:, 2]
        Px_idx  = Cxs[:, 3].long()

        expected = torch.where(Px_idx == 0, Cx_prev + Tx_val, torch.zeros_like(Tx_val))
        is_valid = torch.isclose(Cx_val, expected, rtol=1e-5, atol=1e-8)

        return torch.where(is_valid, torch.zeros_like(Cx_val), torch.full_like(Cx_val, -float("inf")))

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

#Continuous - Resistance Rx | Zx,  Rx = exp(Zx)
class Rx:
    def __init__(self, childs, parents, device="cpu"):
        """
        Rx{t} = Rx{t-1} - exp(Zx{t})  (deterministic temporal update)

        exp(Zx) is always positive, ensuring the degradation subtracted
        each timestep is strictly positive.

        childs  : [Variable Rx]
        parents : [Variable Rx_prev, Variable Zx]
        """
        self.childs = childs
        self.parents = parents
        self.device = device

    # ------------------------------------------------------------------
    def sample(self, Cs_pars):
        """
        Cs_pars : (N, 2)
            Cs_pars[:,0] = Rx_prev value
            Cs_pars[:,1] = Zx value

        Returns
        -------
        Cs   : (N,) Rx = Rx_prev - exp(Zx)
        logp : (N,) zeros  (deterministic node)
        """
        Cs_pars = Cs_pars.to(self.device)

        Rx_prev = Cs_pars[:, 0]
        Zx_val  = Cs_pars[:, 1]

        Cs   = Rx_prev - torch.exp(Zx_val)
        logp = torch.zeros_like(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (N, 3)
            Cs[:,0] = Rx value      (child)
            Cs[:,1] = Rx_prev value (parent 0)
            Cs[:,2] = Zx value      (parent 1)

        Returns
        -------
        log p(Rx | Rx_prev, Zx) : (N,)
            0    where Rx ≈ Rx_prev - exp(Zx)
            -inf otherwise
        """
        Cs = Cs.to(self.device)

        Rx_val  = Cs[:, 0]
        Rx_prev = Cs[:, 1]
        Zx_val  = Cs[:, 2]

        expected   = Rx_prev - torch.exp(Zx_val)
        on_surface = torch.isclose(Rx_val, expected, rtol=1e-5, atol=1e-8)
        return torch.where(on_surface,
                           torch.zeros_like(Rx_val),
                           torch.full_like(Rx_val, -float("inf")))

#Continuous - Initial resistance at t=0 (root, log-normal)
class Rx0:
    def __init__(self, childs, parents=[], mean=0.0, sigma=1.0, device="cpu"):
        """
        Rx0 ~ LogNormal(mean, sigma)  # equivalent to exp(N(mean, sigma^2))

        childs  : [Variable Rx0]
        parents : [empty]
        mean    : mean of the underlying Normal (float)
        sigma   : std of the underlying Normal (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.mean  = float(mean)
        self.sigma = float(sigma)

    # ------------------------------------------------------------------
    def sample(self, n_sample):
        """
        n_sample : int

        Returns
        -------
        Cs   : (n_sample,) sampled Rx0 values
        logp : (n_sample,) log p(Rx0)
        """
        std  = torch.full((n_sample,), self.sigma, device=self.device)
        mean = torch.full_like(std, self.mean)

        dist = LogNormal(mean, std)
        Cs   = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (n_sample,)

        Returns
        -------
        log p(Rx0) : (n_sample,)
        """
        Cs = Cs.to(self.device)

        std  = torch.full_like(Cs, self.sigma)
        mean = torch.full_like(Cs, self.mean)

        dist = LogNormal(mean, std)
        return dist.log_prob(Cs)

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

#Continuous - Latent location factor at t=0 (root marginal)
class Vx0:
    def __init__(self, childs, parents=[], sigma=0.2, device="cpu"):
        """
        Vx{loc}0 ~ Normal(0, sigma^2)   (root initialisation, marginal)

        childs  : [Variable Vx{loc}0]
        parents : [empty]
        sigma   : marginal std (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.sigma = float(sigma)

    # ------------------------------------------------------------------
    def sample(self, n_sample):
        mean = torch.zeros(n_sample, device=self.device)
        std  = torch.full((n_sample,), self.sigma, device=self.device)

        dist = Normal(mean, std)
        Cs = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        Cs = Cs.to(self.device)

        mean = torch.zeros_like(Cs)
        std  = torch.full_like(Cs, self.sigma)

        dist = Normal(mean, std)
        return dist.log_prob(Cs)

#Continuous - Latent location factor | previous timestep (AR(1) via precision matrix)
class Vx:
    def __init__(self, childs, parents, sigma=0.2, rho=0.2, device="cpu"):
        """
        Temporal link Vx{t} | Vx{t-1} derived from a bivariate Gaussian over
        (Vx{t-1}, Vx{t}) with zero mean and covariance

            Sigma = sigma^2 * [[1,   rho],
                               [rho, 1  ]]

        Conditioning via the precision matrix Lambda = Sigma^{-1}:

            mean_{t|t-1} = mu_b - Lambda_bb^{-1} Lambda_ba (Vx{t-1} - mu_a)
            var_{t|t-1}  = Lambda_bb^{-1}

        With zero means this reduces to the AR(1) update
            mean = rho * Vx{t-1},   var = sigma^2 (1 - rho^2)

        so each timestep's prior mean is shifted toward the previous sampled
        value (prior updating through the temporal direction).

        childs  : [Variable Vx{t}]
        parents : [Variable Vx{t-1}]
        sigma   : marginal std of each Vx (float)
        rho     : lag-1 correlation coefficient (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device
        self.sigma = float(sigma)
        self.rho = float(rho)

        # Build covariance, invert to precision, derive conditional coefficients
        var = self.sigma ** 2
        Sigma = torch.tensor([[var,           self.rho * var],
                              [self.rho * var, var          ]], dtype=torch.float64)
        Lambda = torch.inverse(Sigma)
        L_bb = Lambda[1, 1]
        L_ba = Lambda[1, 0]
        L_bb_inv = 1.0 / L_bb

        # mean coefficient on (Vx{t-1} - mu_a); equals rho. var = sigma^2 (1 - rho^2)
        self.b_coef   = float(-L_bb_inv * L_ba)
        self.cond_var = float(L_bb_inv)
        self.cond_std = float(self.cond_var ** 0.5)

    # ------------------------------------------------------------------
    def sample(self, Cs_pars):
        """
        Cs_pars : (N, 1)
            Cs_pars[:,0] = Vx{t-1} value

        Returns
        -------
        Cs   : (N,) Vx{t} ~ Normal(rho * Vx{t-1}, sigma^2 (1 - rho^2))
        logp : (N,) log p(Vx{t} | Vx{t-1})
        """
        Cs_pars = Cs_pars.to(self.device)
        Vx_prev = Cs_pars[:, 0]

        mean = self.b_coef * Vx_prev          # mu_b = 0
        std  = torch.full_like(Vx_prev, self.cond_std)

        dist = Normal(mean, std)
        Cs = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (N, 2)
            Cs[:,0] = Vx{t} value
            Cs[:,1] = Vx{t-1} value

        Returns
        -------
        log p(Vx{t} | Vx{t-1}) : (N,)
        """
        Cs = Cs.to(self.device)
        Vx_val  = Cs[:, 0]
        Vx_prev = Cs[:, 1]

        mean = self.b_coef * Vx_prev
        std  = torch.full_like(Vx_val, self.cond_std)

        dist = Normal(mean, std)
        return dist.log_prob(Vx_val)


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

def define_variables(n_locations=2, n_timesteps=5):
    varis = {}

    for loc in range(1, n_locations + 1):
        # t=0 initialisation (per location): Cx, Vx and Rx
        varis[f'Cx{loc}0'] = variable.Variable(name=f'Cx{loc}0', values=(0, torch.inf))
        varis[f'Vx{loc}0'] = variable.Variable(name=f'Vx{loc}0', values=(-torch.inf, torch.inf))
        varis[f'Rx{loc}0'] = variable.Variable(name=f'Rx{loc}0', values=(0, torch.inf))

        for t in range(1, n_timesteps + 1):
            varis[f'Wx{loc}{t}']  = variable.Variable(name=f'Wx{loc}{t}',  values=(0, torch.inf))
            varis[f'Lx{loc}{t}']  = variable.Variable(name=f'Lx{loc}{t}',  values=(0, torch.inf))
            varis[f'Tx{loc}{t}']  = variable.Variable(name=f'Tx{loc}{t}',  values=(0, torch.inf))
            varis[f'Px{loc}{t}']  = variable.Variable(name=f'Px{loc}{t}',  values=['False', 'True'])
            varis[f'Cx{loc}{t}']  = variable.Variable(name=f'Cx{loc}{t}',  values=(0, torch.inf))
            varis[f'Clx{loc}{t}'] = variable.Variable(name=f'Clx{loc}{t}', values=['False', 'True'])
            varis[f'Rx{loc}{t}']  = variable.Variable(name=f'Rx{loc}{t}',  values=(0, torch.inf))
            varis[f'Zx{loc}{t}']  = variable.Variable(name=f'Zx{loc}{t}',  values=(-torch.inf, torch.inf))
            varis[f'Vx{loc}{t}']  = variable.Variable(name=f'Vx{loc}{t}',  values=(-torch.inf, torch.inf))

    # Ux is shared across locations, one per timestep
    for t in range(1, n_timesteps + 1):
        varis[f'Ux{t}'] = variable.Variable(name=f'Ux{t}', values=(-torch.inf, torch.inf))

    return varis

def define_probs(varis, n_locations=2, n_timesteps=5, vx_sigma=0.2, vx_rho=0.2, device='cpu'):
    probs = {}

    for loc in range(1, n_locations + 1):
        # t=0 initialisation: Cx, Vx and Rx (root marginals)
        probs[f'Cx{loc}0'] = Cx0(childs=[varis[f'Cx{loc}0']], device=device)
        probs[f'Vx{loc}0'] = Vx0(childs=[varis[f'Vx{loc}0']], sigma=vx_sigma, device=device)
        probs[f'Rx{loc}0'] = Rx0(childs=[varis[f'Rx{loc}0']], device=device)

        for t in range(1, n_timesteps + 1):
            probs[f'Wx{loc}{t}']  = Wx(childs=[varis[f'Wx{loc}{t}']], device=device)
            probs[f'Lx{loc}{t}']  = Lx(childs=[varis[f'Lx{loc}{t}']], device=device)
            probs[f'Tx{loc}{t}']  = Tx(childs=[varis[f'Tx{loc}{t}']],
                parents=[varis[f'Wx{loc}{t}'], varis[f'Lx{loc}{t}']],
                device=device,
            )
            probs[f'Px{loc}{t}']  = cpt.Cpt(
                childs=[varis[f'Px{loc}{t}']],
                C=np.array([[0], [1]]),
                p=np.array([1.0, 0.0]),
                device=device,
            )
            # Cx chains temporally: parent is Cx{loc}{t-1} (same location, previo
            # us step)
            probs[f'Cx{loc}{t}']  = Cx(
                childs=[varis[f'Cx{loc}{t}']],
                parents=[varis[f'Cx{loc}{t-1}'], varis[f'Tx{loc}{t}'], varis[f'Px{loc}{t}']],
                device=device,
            )
            # Vx chains temporally: AR(1) link to Vx{loc}{t-1} via precision matrix
            probs[f'Vx{loc}{t}']  = Vx(
                childs=[varis[f'Vx{loc}{t}']],
                parents=[varis[f'Vx{loc}{t-1}']],
                sigma=vx_sigma,
                rho=vx_rho,
                device=device,
            )
            probs[f'Zx{loc}{t}']  = Zx(
                childs=[varis[f'Zx{loc}{t}']],
                parents=[varis[f'Ux{t}'], varis[f'Vx{loc}{t}']],
                device=device,
            )
            # Rx chains temporally: Rx{t} = Rx{t-1} - Zx{t}
            probs[f'Rx{loc}{t}']  = Rx(
                childs=[varis[f'Rx{loc}{t}']],
                parents=[varis[f'Rx{loc}{t-1}'], varis[f'Zx{loc}{t}']],
                device=device,
            )
            probs[f'Clx{loc}{t}'] = Clx(
                childs=[varis[f'Clx{loc}{t}']],
                parents=[varis[f'Cx{loc}{t}'], varis[f'Rx{loc}{t}']],
                device=device,
            )

    # Ux: one per timestep (t>=1), shared across locations
    for t in range(1, n_timesteps + 1):
        probs[f'Ux{t}'] = Ux(childs=[varis[f'Ux{t}']], device=device)

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

    n_locations = 2
    n_timesteps = 5

    # Vx temporal AR(1): marginal std and lag-1 correlation
    vx_sigma = 0.2
    vx_rho   = 0.2

    varis = define_variables(n_locations=n_locations, n_timesteps=n_timesteps)
    probs = define_probs(
        varis,
        n_locations=n_locations,
        n_timesteps=n_timesteps,
        vx_sigma=vx_sigma,
        vx_rho=vx_rho,
        device=device,
    )

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