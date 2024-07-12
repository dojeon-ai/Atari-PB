from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .base import BaseHead
from einops import rearrange, repeat
from src.models.layers import init_normalization, CrossAttentionTransformer, get_2d_sincos_pos_embed


class SiamMAEHead(BaseHead):
    name = 'siammae_head'
    def __init__(self, 
                 obs_shape,
                 in_shape, 
                 action_size,
                 num_heads,
                 norm_type,
                 transformer_decoder):

        super().__init__(in_shape, action_size)
        self.obs_shape = obs_shape
        self.num_heads = num_heads
        self.num_patches = in_shape[0]-1
        self.in_dim = in_shape[1]
        self.patch_size = action_size

        # Linear embedding layer
        embed_dim = transformer_decoder['dim']
        self.embed_linear = nn.Sequential(
            nn.Linear(self.in_dim, embed_dim),
            nn.ReLU(inplace=True)
        )

        # sin-cos position embedding
        pos_embed = get_2d_sincos_pos_embed(
                                embed_dim=embed_dim,
                                grid_size=int(self.num_patches**0.5),
                                cls_token=True,
                                add_token=False
                            )
        self.pos_embed = nn.Parameter(torch.tensor(pos_embed, dtype=torch.float32),
                                      requires_grad=False)
        
        
        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.in_dim))

        # normalization layer
        self.norm = init_normalization(channels=embed_dim, norm_type=norm_type, one_d=True)
        
        # transformer decoder
        self.transformer_decoder = CrossAttentionTransformer(**transformer_decoder)

        # Final game-wise linear layer
        self.recon_w = nn.Embedding(num_heads, embed_dim * self.patch_size)
        self.recon_b = nn.Embedding(num_heads, self.patch_size)
        nn.init.normal_(self.recon_w.weight, std=2.0/((embed_dim+self.patch_size)**0.5))
        nn.init.constant_(self.recon_b.weight, 0)
    
    def nekot_ksam(self, x, recover_idx):
        # Inverse function of mask_token in neck (...sort of)
        # Put tokens back to its original index (before masking),
        # and fill masked (erased) indexes with mask tokens.
        # x: (n p+1 d), p tokens & CLS token
        # recover_idx: (n P), P = # tokens before masking
        # out: (n P+1 d)
        
        # slice CLS token
        cls_token = x[:,:1,:]
        x = x[:,1:,:]

        n, p, d = x.shape
        masks = self.mask_token.expand(n, self.num_patches-p, -1)
        x = torch.cat([x, masks], dim=1)
        row_idx = torch.arange(n).unsqueeze(1)
        x = x[row_idx, recover_idx] # unshuffle

        # reattach CLS token
        x = torch.cat((cls_token, x), dim=1)
        return x

    def forward(self, x, target_x, idx=None, target_mask_info=None):
        """
        [params] x (torch.Tensor: (n, t, p, d))
        [params] target_x (torch.Tensor: (n, t, p', d)) masked
        [params] idx (torch.Tensor: (n, t)) idx of linear prediction to utilize
        [params] recover_idx (torch.Tensor: (n, p)) idx of tokens before masking
        [returns] pred_x (torch.Tensor: (n, t, c, h, w)) reconstruction of target before masking
        """

        # get shape variables
        n, t, p, d = x.shape
        f, c, h, w = self.obs_shape
        np = int(self.num_patches ** 0.5) # num patches in one side
        ph = h // np # patch height, width
        pw = w // np
        assert (h % np == 0) and (w % np == 0)

        if idx is None:
            idx = torch.zeros((n, t), device=x.device).long()
        if target_mask_info is None:
            recover_idx = torch.arange(self.num_patches).repeat(n*t).reshape(n*t,self.num_patches)
        else:
            recover_idx = target_mask_info['recover_idx'] 

        x = rearrange(x, 'n t p d -> (n t) p d')
        x = self.embed_linear(x)
        x = x + self.pos_embed

        target_x = rearrange(target_x, 'n t p d -> (n t) p d')
        target_x = self.embed_linear(target_x)
        target_x = self.nekot_ksam(target_x, recover_idx) # undo mask_token operation
        target_x = target_x + self.pos_embed
        
        x, _ = self.transformer_decoder(target_x, k=x, v=x) # Cross attention
        x = self.norm(x)

        # Game-wise linear
        recon_w = rearrange(self.recon_w(idx), 'n t (d1 d2) -> (n t) d1 d2', d1=self.in_dim, d2=self.patch_size)
        recon_b = rearrange(self.recon_b(idx), 'n t d -> (n t) 1 d')
        x = torch.bmm(x, recon_w) + recon_b
        x = x[:,1:,:] # Remove CLS
        x = F.sigmoid(x)

        # Unpachify
        x = rearrange(x, "n p (f c ph pw) -> n p f c ph pw", f=f, c=c, ph=ph, pw=pw)
        x = rearrange(x, "(n t) (p1 p2) f c ph pw -> n t f c (p1 ph) (p2 pw)", n=n, t=t, p1=np, p2=np)
        info = {}
        
        return x, info