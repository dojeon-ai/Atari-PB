import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import _standard_normal
from einops import rearrange


######################
# transformer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 2, dropout = 0.):
        super().__init__()
        head_dim = dim // heads
        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) 

    def forward(self, x, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'n t (h d) -> n h t d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attn_mask is not None:
            dots.masked_fill_(attn_mask.unsqueeze(1).bool(), -1e4) # -1e4 for amp
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'n h t d -> n t (h d)')
        out = self.to_out(out)
        return out, attn

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 2, dropout = 0.):
        super().__init__()
        head_dim = dim // heads
        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.q_embed = nn.Linear(dim, dim, bias=False)
        self.k_embed = nn.Linear(dim, dim, bias=False)
        self.v_embed = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) 

    def forward(self, x, k, v, attn_mask=None):
        qkv = (self.q_embed(x), self.k_embed(k), self.v_embed(v))
        q, k, v = map(lambda t: rearrange(t, 'n t (h d) -> n h t d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attn_mask is not None:
            dots.masked_fill_(attn_mask.unsqueeze(1).bool(), -1e9)
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'n h t d -> n t (h d)')
        out = self.to_out(out)
        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.apply(transformer_init)

    def forward(self, x, attn_mask=None):
        attn_maps = []
        for attn, ff in self.layers:
            attn_x, attn_map = attn(x, attn_mask=attn_mask)
            x = attn_x + x
            x = ff(x) + x
            attn_maps.append(attn_map)
            
        return x, attn_maps

class CrossAttentionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.apply(transformer_init)

    def forward(self, x, k, v, attn_mask=None):
        attn_maps = []
        for xattn, attn, ff in self.layers:
            # Forward order following SiamMAE
            xattn_x, xattn_map = xattn(x, k=k, v=v, attn_mask=attn_mask)
            x = xattn_x + x
            attn_x, attn_map = attn(x, attn_mask=attn_mask)
            x = attn_x + x
            x = ff(x) + x
            attn_maps.append(attn_map)
            attn_maps.append(xattn_map)
        return x, attn_maps

    
# positional embedding
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, add_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    H = W = grid_size
    grid_size = grid_size + 1  # for additional token

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    # extract add_pos_emb
    D = pos_embed.shape[-1]
    pos_embed = pos_embed.reshape(H + 1, W + 1, D)
    add_pos_embed = pos_embed[-1:, -1:].reshape(1, D)

    # and crop
    pos_embed = pos_embed[:-1, :-1].reshape(H * W, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    if add_token:
        pos_embed = np.concatenate([pos_embed, add_pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
    
###################
# normalization
def renormalize(tensor, first_dim=1):
    # [params] first_dim: starting dimension to normalize the embedding
    eps = 1e-6
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    _max = torch.max(flat_tensor, first_dim, keepdim=True).values
    _min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - _min)/(_max - _min + eps)

    return flat_tensor.view(*tensor.shape)

def init_normalization(channels, norm_type="bn", one_d=False, num_groups=4):
    assert norm_type in ["bn", "bn_nt", "ln", "ln_nt", "gn", None]
    if norm_type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=True, momentum=0.01)
        else:
            return nn.BatchNorm2d(channels, affine=True, momentum=0.01)
        
    elif norm_type == "bn_nt":
        if one_d:
            return nn.BatchNorm1d(channels, affine=False, momentum=0.01)
        else:
            return nn.BatchNorm2d(channels, affine=False, momentum=0.01)
        
    elif norm_type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=True)
        else:
            return nn.GroupNorm(1, channels, affine=True)
    
    elif norm_type == "ln_nt":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=False)
        else:
            return nn.GroupNorm(1, channels, affine=False)
        
    elif norm_type == 'gn':
        return nn.GroupNorm(num_groups, channels, affine=False)
    
    elif norm_type is None:
        return nn.Identity()
    
class ScaleGrad(torch.autograd.Function):
    """Model component to scale gradients back from layer, without affecting
    the forward pass.  Used e.g. in dueling heads DQN models."""

    @staticmethod
    def forward(ctx, tensor, scale):
        """Stores the ``scale`` input to ``ctx`` for application in
        ``backward()``; simply returns the input ``tensor``."""
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Return the ``grad_output`` multiplied by ``ctx.scale``.  Also returns
        a ``None`` as placeholder corresponding to (non-existent) gradient of 
        the input ``scale`` of ``forward()``."""
        return grad_output * ctx.scale, None
    
    
###################
# initialization
def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        gain = 1.0
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.constant_(m.bias.data, 0)
    
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.constant_(m.bias.data, 0)
        
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)    
        nn.init.constant_(m.bias, 0)
    
    return m

def transformer_init(m, std=0.02):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        #nn.init.normal_(m.weight, std)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
        
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)    
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.Parameter):
        nn.init.normal_(m, std)
    
    return m
