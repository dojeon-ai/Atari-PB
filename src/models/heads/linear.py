from abc import *
import torch.nn as nn
from .base import BaseHead


class LinearHead(BaseHead):
    name = 'linear'
    def __init__(self, in_shape, action_size):
        super().__init__(in_shape, action_size)
        
    def forward(self, x):
        info = {}
        return x, info
