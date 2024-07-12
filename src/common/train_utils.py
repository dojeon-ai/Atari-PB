import numpy as np
import math
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from einops import rearrange


def set_global_seeds(seed):
    # torch.backends.cudnn.enabled = False # https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860#h-2-cudagraphs-cause-oom-9
    torch.backends.cudnn.deterministic = True
    # https://huggingface.co/docs/diffusers/v0.9.0/en/optimization/fp16
    torch.backends.cudnn.benchmark = True 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S
    
def get_grad_norm_stats(model):
    grad_norm = []
    stats = {}
    for p in model.parameters():
        if p.grad is not None:
            grad_norm.append(p.grad.detach().data.norm(2))
    grad_norm = torch.stack(grad_norm)
    stats['min_grad_norm'] = torch.min(grad_norm).item()
    stats['mean_grad_norm'] = torch.mean(grad_norm).item()
    stats['max_grad_norm'] = torch.max(grad_norm).item()

    return stats
