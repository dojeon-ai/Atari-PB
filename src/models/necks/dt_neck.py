from abc import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseNeck
from einops import rearrange
from src.models.layers import init_normalization, Transformer, get_1d_sincos_pos_embed_from_grid


class DTNeck(BaseNeck):
    name = 'dt_neck'
    def __init__(self,
                 in_shape, 
                 action_size,
                 num_heads,
                 norm_type,
                 hidden_dims,
                 t_step,
                 transformer_encoder):

        super().__init__(in_shape, action_size)
        
        #######################
        # spatial projector 
        c, h, w = self.in_shape
        self.spatial_embed = nn.Embedding(num_heads, c*h*w) # Pre-MLP embedding
        self.pool = nn.AvgPool2d((h, w))
        self.spatial_norm = init_normalization(channels=c, norm_type=norm_type)
        self.mlp = nn.Sequential(
            nn.Linear(c, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True)
        )
        
        ######################
        # transformer encoder
        embed_dim = transformer_encoder['dim']

        # action & rtg embedder
        self.act_embedder = nn.Embedding(action_size, embed_dim)
        self.rtg_embedder = nn.Linear(1, embed_dim)

        # transformer
        max_len = (t_step * 3) # [R, S, A]
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(max_len))
        self.pos_embed = nn.Parameter(
            torch.tensor(pos_embed, dtype=torch.float32),
            requires_grad=False
        )
        self.transformer_encoder = Transformer(**transformer_encoder)
        self.out_norm = init_normalization(channels=embed_dim, norm_type=norm_type, one_d=True)
        
        
    def forward(self, x, act=None, rtg=None, idx=None):
        """
        [params] x (torch.Tensor: (n, t, c, h, w))
        [params] act (torch.Tensor: (n, t))
        [params] rtg (torch.Tensor: (n, t))
        [params] idx (torch.Tensor: (n, t)) idx of the spatial embedding
        [returns] x (torch.Tensor: (n, t, d))
        """

        n, t, c, h, w = x.shape
        x = rearrange(x, 'n t c h w -> (n t) c h w')
        
        #######################
        # spatial projection
        if idx is None:
            idx = torch.zeros((n, t), device=x.device).long()
        spatial_embed = self.spatial_embed(idx) # Gamewise spatial embedding
        spatial_embed = rearrange(spatial_embed, 'n t (c h w) -> (n t) c h w', c=c, h=h, w=w)
        x = x * spatial_embed
        
        x = self.pool(x)
        x = rearrange(x, '(n t) c 1 1 -> (n t) c', n=n, t=t)
        x = self.spatial_norm(x)
        x = self.mlp(x)
        x = rearrange(x, '(n t) d -> n t d', n=n, t=t)
        
        #######################
        # transformer encoding
        if act is None:
            act = torch.zeros((n, t), device=x.device).long()
        if rtg is None:
            rtg = torch.zeros((n, t), device=x.device).float()
            
        act = self.act_embedder(act)
        rtg = rearrange(rtg, 'n t -> n t 1')
        rtg = self.rtg_embedder(rtg)

        # stack with inter-leaving [g1, o1, a1, g2, ..., gt, ot, at]
        T = 3 * t
        d = x.shape[-1]
        _x = torch.zeros((n, T, d), device=x.device, dtype=x.dtype)
        
        _x[:, torch.arange(t) * 3, :] += rtg
        _x[:, torch.arange(t) * 3 + 1, :] += x
        _x[:, torch.arange(t) * 3 + 2, :] += act
        
        x = _x + self.pos_embed[:T].unsqueeze(0) 
        attn_mask = 1 - torch.ones((n, T, T), device=(x.device)).tril_()
        x, _ = self.transformer_encoder(x, attn_mask) 
        x = self.out_norm(x) # normalization layer
        
        x = x[:, torch.arange(t) * 3 + 1, :]

        info = {}
        return x, info
    
