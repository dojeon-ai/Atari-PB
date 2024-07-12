from abc import *
import torch.nn as nn
import torch


class BaseNeck(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_shape, action_size):
        super().__init__()
        self.in_shape = in_shape
        self.action_size = action_size

    @classmethod
    def get_name(cls):
        return cls.name

    def reset_parameters(self, **kwargs):
        for name, layer in self.named_children():
            modules = [m for m in layer.children()]
            for module in modules:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()