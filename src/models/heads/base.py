from abc import *
import torch.nn as nn
import torch


class BaseHead(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_shape, action_size):
        super().__init__()
        self.in_shape = in_shape
        self.action_size = action_size

    @classmethod
    def get_name(cls):
        return cls.name

