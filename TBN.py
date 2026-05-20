import torch
from torch.distributions import Normal

# Define custom variables and probability distributions

import torch
from torch.distributions import Normal

class Wx:
    def __init__(self, childs, parents, mean=0.0, sigma=1.0, device="cpu"):
        """
        C ~ Normal(mean, sigma^2)

        childs  : [Variable C]
        parents : [empty]
        mean    : fixed mean (float)
        sigma   : fixed noise std (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.mean = float(mean)
        self.sigma = float(sigma)

        # parent variables - Now redudant
#        self.A = parents[0]
#        self.B = parents[1]

        # value lookup tables
#        self.A_values = torch.tensor(
#            self.A.values, dtype=torch.float32, device=device
#        )
#        self.B_values = torch.tensor(
#            self.B.values, dtype=torch.float32, device=device
#        )

    # ------------------------------------------------------------------
    # def sample(self, Cs_pars): - Cs_pars is parent samples - no longer needed - just no. of samples
    def sample(self, N):
        """
        N : int
            number of samples to draw
        
#        Cs_pars : (N, 2)
#            Cs_pars[:,0] = A index
#            Cs_pars[:,1] = B index

        Returns
        -------
        Cs   : (N,) sampled C values
        logp : (N,) log p(C)
        """
#        Cs_pars = Cs_pars.to(self.device).long()

#        A_idx = Cs_pars[:, 0]
#        B_idx = Cs_pars[:, 1]

        mean = torch.full((N,), self.mean, device=self.device)  
        std = torch.full_like(mean, self.sigma, device=self.device)

        dist = Normal(mean, std)
        Cs = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (N, )

        Returns
        -------
        log p(C) : (N,)
        """
        Cs = Cs.to(self.device)

        C_val = Cs
#        A_idx = Cs[:, 1].long()
#        B_idx = Cs[:, 2].long()


        mean = torch.full_like(Cs, self.mean) 
        std = torch.full_like(Cs, self.sigma)

        dist = Normal(mean, std)
        return dist.log_prob(C_val)