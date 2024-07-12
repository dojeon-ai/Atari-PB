from abc import *
import torch.nn as nn
from .base import BaseNeck


class SpatialMLPNeck(BaseNeck):
    name = 'spatial_mlp'
    def __init__(self, in_shape, action_size):
        super().__init__(in_shape, action_size)
        
    def forward(self, x):
        info = {}
        return x, info
    