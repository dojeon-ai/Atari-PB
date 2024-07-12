from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseNeck
from einops import rearrange
from src.models.layers import init_normalization


class MHSpatialMLPNeck(BaseNeck):
    name = 'mh_spatial_mlp'
    def __init__(self, 
                 in_shape, 
                 action_size,
                 num_heads,
                 norm_type,
                 hidden_dims):
        
        super().__init__(in_shape, action_size)
        c, h, w = self.in_shape
        self.spatial_embed = nn.Embedding(num_heads, c*h*w)
        self.pool = nn.AvgPool2d((h, w))
        self.norm = init_normalization(channels=c, norm_type=norm_type)
        self.mlp = nn.Sequential(
            nn.Linear(c, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, idx=None):
        """
        [params] x (torch.Tensor: (n, t, c, h, w))
        [params] idx (torch.Tensor: (n, t)) idx of the spatial embedding
        [returns] x (torch.Tensor: (n, t, d))
        """
        n, t, c, h, w = x.shape
        x = rearrange(x, 'n t c h w -> (n t) c h w')
        
        if idx is None:
            idx = torch.zeros((n, t), device=x.device).long()

        spatial_embed = self.spatial_embed(idx)
        spatial_embed = rearrange(spatial_embed, 'n t (c h w) -> (n t) c h w', c=c, h=h, w=w)
        x = x * spatial_embed

        x = self.pool(x)
        x = rearrange(x, 'n c 1 1 -> n c')
        x = self.norm(x)
        x = self.mlp(x)        
        x = rearrange(x, '(n t) c -> n t c', t=t)
        
        info = {}
        
        return x, info
