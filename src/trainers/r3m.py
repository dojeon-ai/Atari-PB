import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import copy
from einops import rearrange
from .base import BaseTrainer
from src.common.logger import AgentLogger
from src.common.schedulers import LinearScheduler


class R3MTrainer(BaseTrainer):
    name = 'r3m'
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

        # get batch
        game_id = batch['game_id'].contiguous()
        done = batch['done'].contiguous()
        x0 = batch['obs'] # [n t f 1 h w]
        x1, x2 = batch['next_obs'].chunk(2, dim=3) # [n t f 2 h w]
        target_x = torch.cat((x1, x2)) # [2n t f 1 h w]
        target_game_id = torch.cat([game_id]*2)

        # augmentation
        n, t, f, c, h, w = x0.shape
        x0 = rearrange(x0, 'n t f c h w -> n (t f c) h w')
        x0 = self.aug_func(x0)
        x0 = rearrange(x0, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)
        target_x = rearrange(target_x, 'nn t f c h w -> nn (t f c) h w')
        target_x = self.aug_func(target_x)
        target_x = rearrange(target_x, 'nn (t f c) h w -> nn t f c h w', t=t, f=f, c=c)

        # model
        x0, _ = self.model.backbone(x0)
        x0, _ = self.model.neck(x0, game_id)
        x0, _ = self.model.head(x0, game_id)
        x0 = rearrange(x0, 'n t d -> (n t) d')

        with torch.no_grad():
            # includes both positive & negative
            target_x, _ = self.target_backbone(target_x)
            target_x, _ = self.target_neck(target_x, target_game_id)
            target_x = rearrange(target_x, 'nn t d -> (nn t) d')
        
        # Chunk
        x1, x2 = target_x.chunk(2, dim=0) # [n d]
        
        # DDP gather
        if self.cfg.distributed:
            x = torch.stack([x0, x1, x2], dim=1) # [n 3 d]
            x_list = [torch.empty_like(x) for _ in range(self.cfg.num_gpus)]
            done_list = [torch.empty_like(done) for _ in range(self.cfg.num_gpus)]
            game_id_list = [torch.empty_like(game_id) for _ in range(self.cfg.num_gpus)]
            # Share variables for loss computation across all GPUs
            dist.all_gather(tensor=x, tensor_list=x_list)
            dist.all_gather(tensor=done, tensor_list=done_list)
            dist.all_gather(tensor=game_id, tensor_list=game_id_list)
            # all_gather() purely copies the value, and the resulting tensor will have no computational graph.
            # To maintain the gradient graph, each GPU must plaster their own tensor (with computational graph) in the corresponding place.
            x_list[self.logger.rank] = x
            x = torch.cat(x_list, dim=0).requires_grad_() # [N 3 d]
            x0, x1, x2 = x.chunk(3, dim=1) # [N 1 d]
            x0, x1, x2 = x0.squeeze(dim=1), x1.squeeze(dim=1), x2.squeeze(dim=1)
            done = torch.cat(done_list, dim=0)
            game_id = torch.cat(game_id_list, dim=0)   

        # Similarity measure
        def similarity(x,y):
            # Input: Two tensors with shape [n, d]
            # Output: Tensor of similarity between vectors of same index, shape [n, 1]
            return torch.sum(x*y, dim=1, keepdim=True)
            
        def whole_batch_similarity(x,y):
            # Input: Two tensors with shape [n, d]
            # Output: Tensor of similarity between all possible vector pairs, shape [n, n]
            return x @ y.t()

        # within-sample contrastive
        sim_0_1 = similarity(x0,x1)
        sim_0_2 = similarity(x0,x2)
        x0_logits = [sim_0_1, sim_0_2]

        # cross-sample contrastive
        ns = game_id.shape[0]
        mask = torch.eye(ns).to(device=game_id.device) > 0.0
        sim_0_n1 = whole_batch_similarity(x0, x1)
        sim_0_n1 = sim_0_n1.masked_fill_(mask=mask, value=-100) # Mask similarity between same games
        x0_logits.append(sim_0_n1)

        # concat into single matrix
        logits = torch.cat(x0_logits, dim=1) / self.cfg.temperature # temperature

        # loss & acc
        gt_label = torch.zeros(logits.shape[0]).to(self.device).long() # positive = x_1 for both x_0, x_2
        invalid_data = torch.any(done, dim=1) # check done validity
        gt_label[invalid_data] = -100 # assign ignore_index to invalid ones
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, gt_label)
        self_pred_cnt = (torch.argmax(logits, axis=1)==gt_label).float().sum()
        self_pred_acc = self_pred_cnt / float(logits.shape[0] - invalid_data.sum().item())
        
        # logs      
        log_data = {'loss': loss.item(),
                    'self_pred_acc': self_pred_acc.item()}
                    
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