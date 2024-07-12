from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseNeck
from einops import rearrange


class LinearNeck(BaseNeck):
    name = 'linear'
    def __init__(self, 
                 in_shape, 
                 action_size,
                 hidden_dim):
        
        super().__init__(in_shape, action_size)
        c, h, w = self.in_shape
        self.linear = nn.Sequential(
            nn.Linear(c*h*w, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, idx=None):
        """
        [params] x (torch.Tensor: (n, t, c, h, w))
        [params] idx (torch.Tensor: (n, t)) idx of the spatial embedding
        [returns] x (torch.Tensor: (n, t, d))
        """
        n, t, c, h, w = x.shape
        x = rearrange(x, 'n t c h w -> (n t) (c h w)')
        x = self.linear(x)        
        x = rearrange(x, '(n t) c -> n t c', t=t)
        info = {}
        
        return x, info
