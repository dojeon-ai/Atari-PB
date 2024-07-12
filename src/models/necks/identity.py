from abc import *
import torch.nn as nn
from .base import BaseNeck
from einops import rearrange


class IdentityNeck(BaseNeck):
    name = 'identity'
    def __init__(self, in_shape, action_size):
        super().__init__(in_shape, action_size)
        
    def forward(self, x):
        x = rearrange(x, 'n t c h w -> n t (c h w)')
        info = {}
        return x, info