from abc import *
import torch
import torch.nn as nn
from torch.nn import functional as F
from .base import BaseHead
from einops import rearrange


class MHDistributionalHead(BaseHead):
    name = 'mh_distributional'
    def __init__(self, 
                 in_shape, 
                 action_size,
                 num_heads,
                 num_atoms):
        
        super().__init__(in_shape, action_size)
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.in_dim = in_shape[0]
        self.weights = nn.Embedding(num_heads, self.in_dim * action_size * num_atoms)
        self.biases = nn.Embedding(num_heads, action_size * num_atoms)

    def forward(self, x, idx=None):
        """
        [params] x (torch.Tensor: (n, t, d))
        [params] idx (torch.Tensor: (n, t)) idx of each head to utilize
        [returns] x (torch.Tensor: (n, t, a, num_atoms))
        """
        n, t, d = x.shape
        a = self.action_size
        n_a = self.num_atoms
        if idx is None:
            idx = torch.zeros((n, t), device=x.device).long()
        
        weights = self.weights(idx)
        biases = self.biases(idx)
        
        # manually perform 'nn.Linear' operation
        x = rearrange(x, 'n t d -> (n t) 1 d')
        weights = rearrange(weights, 'n t (d a n_a) -> (n t) d (a n_a)', d=d, a=a, n_a=n_a)
        biases = rearrange(biases, 'n t (a n_a) -> (n t) (a n_a)', a=a, n_a=n_a)
        
        x = torch.bmm(x, weights).squeeze(1) + biases
        x = rearrange(x, '(n t) (a n_a) -> n t a n_a', t=t, a=a, n_a=n_a)  
        log_x = F.log_softmax(x, dim=-1)
        x = torch.exp(log_x)
 
        info = {'log': log_x} # keep log value for numerical stability in back-prop
        return x, info

    def get_num_atoms(self):
        return self.num_atoms