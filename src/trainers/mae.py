import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from einops import rearrange
from .base import BaseTrainer
from src.common.logger import AgentLogger

import wandb


class MAETrainer(BaseTrainer):
    name = 'mae'
    def __init__(self,
                 cfg,
                 device,
                 train_loader,
                 train_sampler,
                 eval_loader,
                 eval_sampler,
                 eval_env,
                 logger, 
                 aug_func,
                 model):
        
        super().__init__(cfg, 
                         device, 
                         train_loader, 
                         train_sampler,
                         eval_loader, 
                         eval_sampler, 
                         eval_env,
                         logger, 
                         aug_func, 
                         model)  

        if self.cfg.resume:
            self.start_epoch, self.start_step = self.load_checkpoint()        

    def compute_loss(self, batch):

        x_origin = batch['obs']
        game_id = batch['game_id']
        n, t, f, c, h, w = x_origin.shape

        # Augmentation
        x = copy.deepcopy(x_origin)
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        # Forward
        x, _ = self.model.backbone(x)
        x, mask_info = self.model.neck(x, game_id) # game-wise spatial embedding
        x_recon, _ = self.model.head(x, game_id, mask_info) # game-wise prediction head
        
        # MSE loss
        loss = torch.mean((x_recon - x_origin) ** 2)

        # logs        
        log_data = {'loss': loss.item(),
                    'x': wandb.Image(x_origin[0,0,0]),
                    'x_recon': wandb.Image(x_recon[0,0,0])}
        
        return loss, log_data
    
    def update(self, batch, step):
        raise NotImplementedError
    
    def rollout(self):
        raise NotImplementedError