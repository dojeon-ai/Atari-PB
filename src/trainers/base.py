from abc import *
from typing import Tuple
import numpy as np
import tqdm
import random
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.common.linear_probe import probe_action
from src.common.schedulers import CosineAnnealingWarmupRestarts
from src.common.metrics import *
from src.common.train_utils import get_grad_norm_stats
from src.common.losses import SoftmaxFocalLoss
from src.common.vis_utils import visualize_plot
from einops import rearrange


class BaseTrainer():
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
        
        super().__init__()

        self.cfg = cfg  
        self.device = device
        self.eval_env = eval_env
        self.logger = logger
        self.model = model.to(self.device)
        self.start_epoch = 1
        self.start_step = 0
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.eval_loader = eval_loader
        self.eval_sampler = eval_sampler
        
        # Training related instances
        if self.cfg.num_epochs > 0 and self.train_loader is not None:
            self.aug_func = aug_func.to(self.device)
            self.optimizer = self._build_optimizer(cfg.optimizer)
            self.lr_scheduler = self._build_scheduler(self.optimizer, cfg.scheduler)
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)

        self.distributed = self.cfg.distributed
        if self.distributed:            
            self.model.backbone = torch.nn.parallel.DistributedDataParallel(self.model.backbone)
            self.model.head = torch.nn.parallel.DistributedDataParallel(self.model.head)
            self.model.neck = torch.nn.parallel.DistributedDataParallel(self.model.neck)
                        
        if self.cfg.compile:
            self.model.backbone = torch.compile(self.model.backbone)
            self.model.head = torch.compile(self.model.head)
            self.model.neck = torch.compile(self.model.neck)
                
    @classmethod
    def get_name(cls):
        return cls.name        

    def _build_optimizer(self, optimizer_cfg):
        optimizer_type = optimizer_cfg.pop('type')
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), 
                              **optimizer_cfg)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), 
                              **optimizer_cfg)
        else:
            raise ValueError

    def _build_scheduler(self, optimizer, scheduler_cfg):
        first_cycle_steps = len(self.train_loader) * self.cfg.num_epochs
        return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                             first_cycle_steps=first_cycle_steps,
                                             **scheduler_cfg)
 
    @abstractmethod
    def compute_loss(self, batch) -> Tuple[torch.Tensor, dict]:
        pass
    
    @abstractmethod
    # custom model update other than backpropagation (e.g., ema)
    def update(self, batch, step):
        pass
    
    # The reason why collate_fn is inside the trainer function.
    # DDP: False, num_workers: 0 -> o
    # DDP: False, num_workers> 0 -> o
    # DDP: True,  num_workers: 0 -> o
    # DDP: True,  num_workers> 0 -> x
    # Weiredly, we found that DDP with num_workers > 0 does not allow to pass collate_fn to its child.
    # We failed to trouble shoot and decided to collate the batch inside the trainer.
    def _collate(self, batch):
        """
        [params] 
            observation: (n, t+f-1, c, h, w) 
            next_observation: (n, t+f-1, c, h, w)
            action:   (n, t+f-1)
            reward:   (n, t+f-1)
            terminal: (n, n+t+f-1) * different n's (batch vs n_step)
            rtg:      (n, t+f-1)
            game_id:  (n, t+f-1)            
        [returns] 
            (c = 1 in ATARI)
            obs:      (n, t, f, c, h, w) 
            next_obs: (n, t, f, c, h, w)
            action:   (n, t)
            reward:   (n, t)
            done:     (n, n+t)
            rtg:      (n, t)
            game_id:  (n, t)    
        """
        f = self.cfg.frame
        t = self.cfg.t_step
        obs = batch['observation']
        action = batch['action']
        reward = batch['reward']
        done = batch['terminal']
        rtg = batch['rtg']
        game_id = batch['game_id']
        next_obs = batch['next_observation']

        # process data-format
        obs = rearrange(obs, 'n tf c h w -> n tf 1 c h w')
        obs = obs.repeat(1, 1, f, 1, 1, 1)
        next_obs = rearrange(next_obs, 'n tf c h w -> n tf 1 c h w')
        next_obs = next_obs.repeat(1, 1, f, 1, 1, 1)
        action = action.long()
        reward = torch.nan_to_num(reward).sign()
        done = done.bool()
        rtg = rtg.float()
        game_id = game_id.long()

        # frame-stack
        if f != 1:
            for i in range(1, f):
                obs[:, :, i] = obs[:, :, i].roll(-i, 1)
                next_obs[:, :, i] = next_obs[:, :, i].roll(-i, 1)
            obs = obs[:, :-(f-1)]
            next_obs = next_obs[:, :-(f-1)]
            action = action[:, f-1:]
            reward = reward[:, f-1:]
            done = done[:, f-1:]
            rtg = rtg[:, f-1:]
            game_id = game_id[:, f-1:]
            
        # lazy frame to float
        obs = obs / 255.0
        next_obs = next_obs / 255.0
            
        batch = {
            'obs': obs,
            'next_obs': next_obs,
            'act': action,
            'rew': reward,
            'done': done,
            'rtg': rtg,
            'game_id': game_id,                            
        }            
            
        return batch

    def train(self):
        step = self.start_step
        
        if step == 0 and (self.cfg.eval_every != -1):
           # initial evaluation
            self.model.eval()
            eval_logs = {}
            eval_logs = self.evaluate()
            eval_logs['epoch'] = 0
            self.logger.update_log(**eval_logs)
            self.logger.write_log(step)
        
        # train
        use_amp = self.cfg.use_amp
        for epoch in range(self.start_epoch, self.cfg.num_epochs+1):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
                
            for batch in tqdm.tqdm(self.train_loader):   
                # forward
                self.model.train()
            
                for key, value in batch.items():
                    batch[key] = value.to(self.device)
                batch = self._collate(batch)
            
                self.optimizer.zero_grad()    
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, train_logs = self.compute_loss(batch)
        
                # backward
                self.scaler.scale(loss).backward()

                # gradient clipping
                # unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                if self.cfg.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
                # perform custom update function
                if (step % self.cfg.target_update_every == 0) and (self.cfg.target_update_every != -1):
                    self.update(batch, step)

                # log        
                grad_stats = get_grad_norm_stats(self.model)
                scheduler_logs = {}
                scheduler_logs['learning_rate'] = self.lr_scheduler.get_lr()[0]
                if hasattr(self, 'tau_scheduler'):
                    scheduler_logs['momentum_update_ratio'] = self.tau_scheduler.get_value(step)
                train_logs.update(grad_stats)
                train_logs.update(scheduler_logs)

                self.logger.update_log(**train_logs)
                if step % self.cfg.log_every == 0:
                    self.logger.write_log(step)
                    
            
                # proceed
                self.lr_scheduler.step()
                step += 1

            torch.cuda.empty_cache()

            # Checkpoint save
            if (epoch % self.cfg.save_every == 0) and (self.cfg.save_every != -1):
                self.save_checkpoint(epoch, step)
            if self.cfg.save_every != -1:
                self.save_checkpoint(epoch, step, name='latest') # Constantly overwrite latest epoch

            # Eval log
            epoch_log = {'epoch': epoch}
            self.logger.update_log(**epoch_log)
            
            
            if (epoch % self.cfg.eval_every == 0) and (self.cfg.eval_every != -1):
                self.model.eval()
                eval_logs = self.evaluate()
                self.logger.update_log(**eval_logs)
            
            if (epoch % self.cfg.rollout_every == 0) and (self.cfg.rollout_every != -1):
                self.model.eval()
                rollout_logs = self.rollout()
                self.logger.update_log(**rollout_logs)
            
            self.logger.write_log(step)

                
    def _encode_features_for_probing(self):
        _stack_of_batches = {}
        for batch in tqdm.tqdm(self.eval_loader):  
            for key, value in batch.items():
                batch[key] = value.to(self.device)
            batch = self._collate(batch)
            
            obs = batch['obs']
            game_id = batch['game_id']
            with torch.no_grad():
                feat, _ = self.model.backbone(obs)
                feat, _ = self.model.neck(feat, idx=game_id) # game-wise spatial embedding         

            batch['obs'] = feat # do not stack obs (memory-cost)
            for key, value in batch.items():
                if key not in _stack_of_batches:
                    _stack_of_batches[key] = []
                _stack_of_batches[key].append(value)
    
    
    def evaluate(self) -> dict:
        eval_logs = {}
        
        # encode features for probing
        # do not stack obs (memory overhead)   
        _stack_of_batches = {}
        for batch in tqdm.tqdm(self.eval_loader):  
            for key, value in batch.items():
                batch[key] = value.to(self.device)
            batch = self._collate(batch)
            obs = batch['obs']
            game_id = batch['game_id']
            with torch.no_grad():
                backbone_feat, _ = self.model.backbone(obs)
                neck_feat, _ = self.model.neck(backbone_feat, game_id) # game-wise embedding   
            
            batch['obs_neck'] = neck_feat
            del batch['obs']
            del batch['next_obs']
            
            for key, value in batch.items():
                if key not in _stack_of_batches:
                    _stack_of_batches[key] = []
                _stack_of_batches[key].append(value.cpu())
                
        stack_of_batches = {}
        for key, value in _stack_of_batches.items():
            _value = torch.cat(value)
            # Squeeze 't' dimension (if not squeezed by the model) (not a good method, but it does the job)
            _value = _value.flatten(start_dim=0, end_dim=1).unsqueeze(1)
            stack_of_batches[key] = _value
        
        # perform linear probing
        neck_feats = stack_of_batches['obs_neck']
        acts = stack_of_batches['act']
        ids = stack_of_batches['game_id']
        action_size = self.cfg.action_size
        act_acc = probe_action(neck_feats, acts, ids, action_size, self.device)
        
        # features to estimate feat.rank and dormant neuron
        if len(neck_feats) < self.cfg.eval_data_size:
            rand_idxs = np.arange(len(neck_feats))
        else:
            rand_idxs = np.random.choice(len(neck_feats), size=self.cfg.eval_data_size, replace=False)
        neck_feats = neck_feats[rand_idxs, 0]
        
        # dormant neurons
        neck_dormant_rate = dormant_neurons(neck_feats)

        # feature rank
        neck_feat_rank, eigen_vals_n = rankme(neck_feats)
        
        # plot eigenvalue spectrum
        x_n = np.arange(1, len(eigen_vals_n)+1)
        eigen_spectrum_n = visualize_plot(
            x=x_n, y=eigen_vals_n, 
            x_label='Index', y_label='Eigenvalue', title='Neck Eigenvalue Spectrum'
        )
        log_eigen_spectrum_n = visualize_plot(
            x=x_n, y=eigen_vals_n.log(), 
            x_label='Index', y_label='Log Eigenvalue', title='Neck Eigenvalue Spectrum'
        )
        
        del _stack_of_batches
        del stack_of_batches
        
        eval_logs = {
            'eval_act_acc': act_acc,
            'neck_dormant_rate': neck_dormant_rate,
            'neck_feat_rank': neck_feat_rank,
            'neck_eigen_spectrum': wandb.Image(eigen_spectrum_n),
            'log_neck_eigen_spectrum': wandb.Image(log_eigen_spectrum_n)
        }
        
        return eval_logs
        
    @abstractmethod
    def rollout(self) -> dict:
        pass

    # Override below two functions if your checkpoint has different elements (e.g. momentum encoder)
    def save_checkpoint(self, epoch, step, name=None):
        # Save only once at rank 0
        if self.logger.rank == 0:
            if name is None:
                name = 'epoch'+str(epoch)
            save_dict = {'backbone': self.model.backbone.state_dict(),
                          'neck': self.model.neck.state_dict(),
                          'head': self.model.head.state_dict(),
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
                     'optimizer': self.optimizer,
                     'lr_scheduler': self.lr_scheduler,
                     'scaler': self.scaler,
                     'epoch': -1,
                     'step': -1,}

        ret = self.logger.load_dict(load_dict=load_dict,
                                    path=self.cfg.ckpt_path,
                                    device=self.device)
        start_epoch = ret['epoch']+1
        start_step = ret['step']+1

        print('Starting at epoch', start_epoch)
        return start_epoch, start_step


