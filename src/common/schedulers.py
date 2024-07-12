import torch
import torch.nn as nn
import numpy as np
import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import Tuple
from abc import *


class BaseScheduler(metaclass=ABCMeta):
    def __init__(self, initial_value, final_value, max_step):
        """
        [params] initial_value (float) initial output value
        [params] final_value (float) final output value
        [params] step_size (int) number of timesteps to lineary anneal initial value to the final value
        """
        super().__init__(initial_value, final_value, max_step)

    @classmethod
    def get_name(cls):
        return cls.name

    def get_value(self, step) -> float:
        pass


class LinearScheduler(BaseScheduler):
    name = 'linear'
    def __init__(self, initial_value, final_value, max_step, **kwargs):
        """
        Linear interpolation between initial_value to the final_value.
        """
        self.initial_value = initial_value
        self.final_value   = final_value
        if isinstance(max_step, str):
            max_step = eval(max_step)
        self.max_step = max_step
        self.interval = (final_value - initial_value) / max_step
        
    def get_value(self, step):
        step = min(step, self.max_step)
        
        return self.initial_value + self.interval * step  

class ExponentialScheduler(BaseScheduler):
    name = 'exponential'
    def __init__(self, initial_value, final_value, max_step, reverse=False, **kwargs):
        """
        Exponential interpolation between initial_value to the final_value.
        Args:
            initial_value: float, (0, 1)
            final_value: float, (0, 1)
            reverse: bool, whether to treat 1 as the asmpytote instead of 0.
        """
        self.initial_value = initial_value
        self.final_value   = final_value
        if isinstance(max_step, str):
            max_step = eval(max_step)
        self.max_step = max_step
        self.reverse = reverse
        if self.reverse:
            self.initial_value = 1 - initial_value
            self.final_value = 1 - final_value
        self.interval = (np.log(self.final_value) - np.log(self.initial_value)) / max_step
        
    def get_value(self, step):
        step = min(step, self.max_step)
        start = np.log(self.initial_value)
        end = np.log(self.final_value)
        
        if self.reverse:
            return 1 - np.exp(start + self.interval * step)
        else:
            return np.exp(start + self.interval * step)
        
class CosineScheduler(BaseScheduler):
    name = 'cosine'
    def __init__(self, initial_value, final_value, max_step, **kwargs):
        """
        Cosine interpolation between initial_value to the final_value.
        """
        self.initial_value = initial_value
        self.final_value = final_value
        if isinstance(max_step, str):
            max_step = eval(max_step)
        self.max_step = max_step

    def get_value(self, step):
        step = min(step, self.max_step)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.max_step))
        
        return self.final_value + (self.initial_value - self.final_value) * cosine_decay


##########################
# Optimizers' Schedulers
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr_ratio : float = 0.1,
                 warmup_ratio : float = 0.2,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = max_lr * min_lr_ratio # min learning rate
        self.warmup_steps = int(warmup_ratio * first_cycle_steps) # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
