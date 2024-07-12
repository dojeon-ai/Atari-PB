from abc import *
import torch.nn as nn
import torch
EPS = 1e-5

class BaseBackbone(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_shape, action_size):
        super().__init__()
        self.in_shape = in_shape
        self.action_size = action_size

    @classmethod
    def get_name(cls):
        return cls.name

    @abstractmethod
    def forward(self, x):
        """
        [param] x (torch.Tensor): (n, t, c, h, w)
        [return] x (torch.Tensor): (n, t, d)
        """
        pass
    
    @property
    def output_dim(self):
        pass

    def reset_parameters(self):
        for name, layer in self.named_children():
            modules = [m for m in layer.children()]
            for module in modules:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
