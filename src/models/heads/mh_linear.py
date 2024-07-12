from abc import *
import torch
import torch.nn as nn
from .base import BaseHead
from einops import rearrange


class MHLinearHead(BaseHead):
    name = 'mh_linear'
    def __init__(self, 
                 in_shape,
                 action_size,
                 num_heads):
        
        super().__init__(in_shape, action_size)
        self.num_heads = num_heads
        self.in_dim = in_shape if isinstance(in_shape, int) else in_shape[0]
        self.weights = nn.Embedding(num_heads, self.in_dim * action_size)
        self.biases = nn.Embedding(num_heads, action_size)

    def forward(self, x, idx=None):
        """
        [params] x (torch.Tensor: (n, t, d))
        [params] idx (torch.Tensor: (n, t)) idx of each head to utilize
        [returns] x (torch.Tensor: (n, t, a))
        """
        n, t, d = x.shape
        a = self.action_size
        if idx is None:
            idx = torch.zeros((n, t), device=x.device).long()
        
        weights = self.weights(idx)
        biases = self.biases(idx)
        
        # manually perform 'nn.Linear' operation
        x = rearrange(x, 'n t d -> (n t) 1 d')
        weights = rearrange(weights, 'n t (d a) -> (n t) d a', d=d, a=a)
        biases = rearrange(biases, 'n t a -> (n t) a')
                
        x = torch.bmm(x, weights).squeeze(1) + biases
        x = rearrange(x, '(n t) a -> n t a', t=t)   
 
        info = {}
        return x, info
