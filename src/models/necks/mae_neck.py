from abc import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseNeck
from einops import rearrange
from src.models.layers import init_normalization, Transformer, get_2d_sincos_pos_embed


class MAENeck(BaseNeck):
    name = 'mae_neck'
    def __init__(self,
                 in_shape, 
                 action_size,
                 num_heads,
                 norm_type,
                 hidden_dims,
                 mask_ratio,
                 transformer_encoder):

        super().__init__(in_shape, action_size)
        c, h, w = self.in_shape
        self.mask_ratio = mask_ratio
        self.spatial_embed = nn.Embedding(num_heads, c*h*w) # Pre-MLP embedding
        embed_dim = transformer_encoder['dim']

        pos_embed = get_2d_sincos_pos_embed(
                                embed_dim=embed_dim,
                                grid_size=h,
                                cls_token=True,
                                add_token=False
                            )
        self.pos_embed = nn.Parameter(torch.tensor(pos_embed[1:], dtype=torch.float32),
                                      requires_grad=False) # Transformer pos embedding
        self.cls_token = nn.Parameter(torch.tensor(pos_embed[0], dtype=torch.float32).reshape(1,1,-1)) # Initialize with CLS position embedding

        self.norm = init_normalization(channels=embed_dim, norm_type=norm_type, one_d=True)
        self.mlp = nn.Sequential(
            nn.Linear(c, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True)
        )
        self.transformer_encoder = Transformer(**transformer_encoder)

    def mask_token(self, x, mask_ratio):
        # Randomly mask tokens. Each image is left with L*(1-mask_ratio) tokens.
        N, L, D = x.shape
        num_kept = int(L * (1.0-mask_ratio))

        diceroll = torch.rand(N,L)
        _, shuffled_idx = diceroll.sort(dim=-1)

        if num_kept == L:
            # do not shuffle when no tokens are masked,
            # since they won't be unshuffled if it was eval mode
            shuffled_idx, _ = shuffled_idx.sort(dim=-1)

        keep_idx = shuffled_idx[:,:num_kept] # index of kept tokens
        _, recover_idx = shuffled_idx.sort(dim=-1) # used to put tokens back to original index
        
        row_idx = torch.arange(N).unsqueeze(1)
        x = x[row_idx, keep_idx]
        # mask = torch.zeros(N,L).bool()
        # mask[row_idx, keep_idx] = True
        
        return x, recover_idx
        
    def forward(self, x, idx=None):
        """
        [params] x (torch.Tensor: (n, t, c, h, w))
        [params] idx (torch.Tensor: (n, t)) idx of the spatial embedding
        [returns] x (torch.Tensor: (n*t, p, d)), p = h*w*(1-mask_ratio)
        """

        n, t, c, h, w = x.shape
        x = rearrange(x, 'n t c h w -> (n t) c h w')
        if idx is None:
            idx = torch.zeros((n, t), device=x.device).long()
            mask_ratio = 0.0
        else:
            mask_ratio = self.mask_ratio if self.training else 0.0

        # MLP
        spatial_embed = self.spatial_embed(idx) # Gamewise spatial embedding
        spatial_embed = rearrange(spatial_embed, 'n t (c h w) -> (n t) c h w', c=c, h=h, w=w)
        x = x * spatial_embed
        x = rearrange(x, 'n c h w -> n (h w) c') # Flatten
        x = self.mlp(x)
        
        # Transformer
        x = x + self.pos_embed # Position embedding
        x, recover_idx = self.mask_token(x, mask_ratio) # mask
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) # CLS token
        x, _ = self.transformer_encoder(x) # transformer
        x = self.norm(x) # normalization layer
        x = rearrange(x, '(n t) p d -> n t p d', n=n, t=t)

        if not self.training:
            x = x[:,:,0,:] # CLS token as output

        info = {'recover_idx': recover_idx}
        return x, info
