import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from einops import rearrange
from .base import BaseTrainer
from src.common.logger import AgentLogger


class IDMTrainer(BaseTrainer):
    name = 'idm'
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
        curr_x = batch['obs']
        future_x = batch['next_obs']
        act = batch['act']
        done = batch['done']
        game_id = batch['game_id']
        
        n, t, f, c, h, w = curr_x.shape

        # augmentation
        curr_x = rearrange(curr_x, 'n t f c h w -> n (t f c) h w')
        curr_x = self.aug_func(curr_x)
        curr_x = rearrange(curr_x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)
        future_x = rearrange(future_x, 'n t f c h w -> n (t f c) h w')
        future_x = self.aug_func(future_x)
        future_x = rearrange(future_x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        x = torch.cat([curr_x, future_x], dim=0)
        stacked_game_id = torch.cat([game_id, game_id], dim=0)
        x, _ = self.model.backbone(x)
        x, _ = self.model.neck(x, stacked_game_id) # game-wise spatial embedding
        x = torch.cat(x.chunk(2), dim=2) # concatenate current and future x embeddings ([2n,t,d] -> [n,t,2d])
        act_pred_logits, _ = self.model.head(x, game_id) # game-wise prediction head

        # loss
        act_pred_logits = rearrange(act_pred_logits, 'n t d -> (n t) d')
        act_gt = rearrange(act, 'n t -> (n t)') # t=1
        invalid_data = torch.any(done, dim=1) # check done validity
        act_gt[invalid_data] = -100 # assign ignore_index to invalid ones
        criterion = nn.CrossEntropyLoss()
        loss = criterion(act_pred_logits, act_gt)
        act_pred_cnt = (torch.argmax(act_pred_logits, axis=1)==act_gt).sum()
        act_acc = act_pred_cnt.float() / float(n - invalid_data.sum().item())

        # logs        
        log_data = {'loss': loss.item(),
                    'act_acc': act_acc.item()}
        
        return loss, log_data
    
    def update(self, batch, step):
        raise NotImplementedError
    
    def rollout(self):
        raise NotImplementedError