import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from einops import rearrange
from .base import BaseTrainer
from src.common.logger import AgentLogger
from src.common.schedulers import LinearScheduler
import wandb

class SiamMAETrainer(BaseTrainer):
    name = 'siammae'
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

        x = batch['obs']
        target_x_origin = batch['next_obs']
        game_id = batch['game_id']
        n, t, f, c, h, w = x.shape

        image_log_data = {'x': wandb.Image(x[0,0,0]),
                          'target_x': wandb.Image(target_x_origin[0,0,0])}

        # Augmentation
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)
        target_x = copy.deepcopy(target_x_origin)
        target_x = rearrange(target_x, 'n t f c h w -> n (t f c) h w')
        target_x = self.aug_func(target_x)
        target_x = rearrange(target_x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        # Forward
        xtx = torch.cat((x,target_x))
        xtx, _ = self.model.backbone(xtx)
        x, target_x = xtx.chunk(2)
        x, _ = self.model.neck(x, game_id, is_target=False)
        target_x, mask_info = self.model.neck(target_x, game_id, is_target=True)
        recon_x, _ = self.model.head(x, target_x, game_id, mask_info)

        # MSE loss
        loss = torch.mean((recon_x - target_x_origin) ** 2)
        
        # logs        
        log_data = {'loss': loss.item(),
                    'recon_x': wandb.Image(recon_x[0,0,0])}
        log_data.update(image_log_data)
        
        return loss, log_data
    
    def rollout(self):
        raise NotImplementedError