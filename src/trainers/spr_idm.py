import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from einops import rearrange
from .base import BaseTrainer
from src.common.logger import AgentLogger
from src.common.schedulers import LinearScheduler

class SPRIDMTrainer(BaseTrainer):
    name = 'spr_idm'
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
    
        # Remember to deepcopy PRIOR to compile/DDP, DDP seems to hate copying from already compiled models
        self.target_backbone = copy.deepcopy(model.backbone).to(device)
        self.target_neck = copy.deepcopy(model.neck).to(device)

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

        total_steps = len(self.train_loader) * self.cfg.num_epochs
        cfg.tau_scheduler.max_step = total_steps
        self.tau_scheduler = LinearScheduler(**cfg.tau_scheduler)  

        if self.cfg.compile:
            self.target_backbone = torch.compile(self.target_backbone)
            self.target_neck = torch.compile(self.target_neck)

        if self.cfg.resume:
            self.start_epoch, self.start_step = self.load_checkpoint()  

    def compute_loss(self, batch):

        x = batch['obs']
        act = batch['act']
        game_id = batch['game_id']
        done = batch['done'].type(torch.int64)

        n, t, f, c, h, w = x.shape
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        target_x = x[:,1:]
        pred_x = x[:,:1]

        # online encoder
        pred_x, _ = self.model.backbone(pred_x)
        pred_x, _ = self.model.neck(pred_x, game_id[:,:1])

        norm_x = torch.linalg.vector_norm(pred_x.detach().squeeze(), ord=2, dim=1)
        var_norm_x, mean_norm_x = torch.var_mean(norm_x)

        # momentum encoder
        with torch.no_grad():
            target_x, _ = self.target_backbone(target_x)
            target_x, _ = self.target_neck(target_x, game_id[:,1:])
            
            norm_target_x = torch.linalg.vector_norm(target_x.detach().flatten(start_dim=0, end_dim=1), ord=2, dim=1)
            var_norm_target_x, mean_norm_target_x = torch.var_mean(norm_target_x)

        # prediction
        spr_pred, info = self.model.head(pred_x, target_x, act[:,:-1], game_id[:,:-1])
        spr_pred = rearrange(spr_pred, 'n t d -> (n t) d')
        idm_pred = rearrange(info['idm_pred'], 'n t a -> (n t) a')
        target_x = rearrange(target_x, 'n t d -> (n t) d')

        # SPR loss
        spr_criterion = nn.CosineSimilarity(dim=1)
        spr_loss = -1*spr_criterion(spr_pred, target_x).mean()

        # IDM loss
        gt_label = rearrange(act[:,:-1], 'n t -> (n t)')
        idm_criterion = nn.CrossEntropyLoss()
        idm_loss = idm_criterion(idm_pred, gt_label)
        idm_pred_cnt = (torch.argmax(idm_pred, axis=1)==gt_label).sum()
        idm_acc = idm_pred_cnt.float() / float(n*(t-1))

        # Total loss
        loss = self.cfg.spr_weight*spr_loss + self.cfg.idm_weight*idm_loss

        log_data = {'loss': loss.item(),
                    'spr_loss': spr_loss.item(),
                    'idm_loss': idm_loss.item(),
                    'idm_acc': idm_acc.item()}

        return loss, log_data

    def update(self, batch, step):
        # EMA
        tau = self.tau_scheduler.get_value(step)
        for online, target in zip(self.model.backbone.parameters(), self.target_backbone.parameters()):
            target.data = tau*target.data + (1-tau)*online.data
        for online, target in zip(self.model.neck.parameters(), self.target_neck.parameters()):
            target.data = tau*target.data + (1-tau)*online.data

    def rollout(self):
        raise NotImplementedError
    
    def save_checkpoint(self, epoch, step, name=None):
        if self.logger.rank == 0:
            if name is None:
                name = 'epoch'+str(epoch)
            save_dict = {'backbone': self.model.backbone.state_dict(),
                         'neck': self.model.neck.state_dict(),
                         'head': self.model.head.state_dict(),
                         'target_backbone': self.target_backbone.state_dict(),
                         'target_neck': self.target_neck.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'lr_scheduler': self.lr_scheduler.state_dict(),
                         'scaler': self.scaler.state_dict(),
                         'epoch': epoch,
                         'step': step,}
            self.logger.save_dict(save_dict=save_dict,
                                  name=name)

    def load_checkpoint(self):
        # Dict of elements to load
        load_dict = {'backbone': self.model.backbone,
                     'neck': self.model.neck,
                     'head': self.model.head,
                     'target_backbone': self.target_backbone,
                     'target_neck': self.target_neck,    
                     'optimizer': self.optimizer,
                     'lr_scheduler': self.lr_scheduler,
                     'scaler': self.scaler,
                     'epoch': -1,
                     'step': -1,}

        ret = self.logger.load_dict(path=self.cfg.ckpt_path,
                                    load_dict=load_dict,
                                    device=self.device)
        start_epoch = ret['epoch']+1
        start_step = ret['step']+1
        
        print('Starting at epoch', start_epoch)
        return start_epoch, start_step