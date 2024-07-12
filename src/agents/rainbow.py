from .base import BaseAgent
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy


class RAINBOW(BaseAgent):
    name = 'rainbow'
    def __init__(self,
                 cfg,
                 device,
                 train_env,
                 eval_env,
                 logger, 
                 buffer,
                 aug_func,
                 model):
        
        super().__init__(cfg, device, train_env, eval_env, logger, buffer, aug_func, model)  
        self.target_model = copy.deepcopy(self.model).to(self.device)   
        self.target_model.load_state_dict(self.model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False

        # distributional
        self.num_atoms = self.model.head.num_atoms
        self.v_min = self.cfg.v_min
        self.v_max = self.cfg.v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        if cfg.compile:
            self._compile()
        
    def _compile(self):
        self.model.backbone = torch.compile(self.model.backbone)
        self.model.neck = torch.compile(self.model.neck)
        self.model.head = torch.compile(self.model.head)
        
        self.target_model.backbone = torch.compile(self.target_model.backbone)
        self.target_model.neck = torch.compile(self.target_model.neck)
        self.target_model.head = torch.compile(self.target_model.head)

    def predict(self, model, obs, eps) -> torch.Tensor:
        """
        [params] obs: torch.Tensor (n, t, f, c, h, w) 
        [returns] action: numpy array (n, t)
        """
        # forward
        n, t, f, c, h, w = obs.shape
        game_id = torch.LongTensor([[self.game_id]]).to(self.device)
        game_id = game_id.repeat(n, 1)
        x, _ = model.backbone(obs)
        x, _ = model.neck(x, game_id) # game-wise spatial embedding
        q_dist, _ = model.head(x, game_id) # game-wise prediction head
        
        # distribution -> value
        support = rearrange(self.support, 'n_a -> 1 1 1 n_a')
        q_value = (q_dist * support).sum(-1)
        argmax_action = torch.argmax(q_value, -1).cpu().numpy()
        
        # eps-greedy
        prob = np.random.rand(n, t)
        is_rand = (prob <= eps)
        rand_action = np.random.randint(0, self.cfg.action_size-1, (n, t))
        action = is_rand * rand_action + (1-is_rand) * argmax_action
        
        return action
    
    def forward(self, online_model, target_model, batch, mode, reduction='mean'):
        # get samples from buffer
        obs= batch['obs']
        act = batch['act']
        rew = batch['rew']
        done = batch['done']
        next_obs = batch['next_obs']
        G = batch['G']
        
        n_step = batch['n_step']
        gamma = batch['gamma']
        
        tree_idxs = batch['tree_idxs']
        weights = batch['weights']
        
        # augment the observation if needed
        n, t, f, c, h, w = obs.shape
        obs = rearrange(obs, 'n t f c h w -> n (t f c) h w')
        next_obs = rearrange(next_obs, 'n t f c h w -> n (t f c) h w')
        obs = self.aug_func(obs)
        if self.cfg.aug_target:
            next_obs = self.aug_func(next_obs)
        obs = rearrange(obs, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)
        next_obs = rearrange(next_obs, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        # Calculate current state's q-value distribution
        game_id = torch.LongTensor([[self.game_id]]).to(self.device)
        game_id = game_id.repeat(n, 1)
        x, _ = online_model.backbone(obs)
        x, _ = online_model.neck(x, game_id) # game-wise spatial embedding
        _, head_info = online_model.head(x, game_id) # game-wise prediction head
        log_pred_q_dist = rearrange(head_info['log'], 'n t d n_a -> (n t) d n_a')
        # q_dist = rearrange(q_dist, 'n t d n_a -> (n t) d n_a')
        
        # gather action
        act_idx = act.reshape(-1,1,1).repeat(1,1,self.num_atoms)
        log_pred_q_dist = (log_pred_q_dist).gather(1, act_idx).squeeze(1) # (n, n_a)
        
        with torch.no_grad():
            # Calculate n-th next state's q-value distribution
            # next_target_q_dist: (n, a, num_atoms)
            # target_q_dist: (n, num_atoms)
            ntx, _ = target_model.backbone(next_obs)
            ntx, _ = target_model.neck(ntx, game_id)
            next_target_q_dist, _ = target_model.head(ntx, game_id)
            next_target_q_dist = rearrange(next_target_q_dist, 'n t d n_a -> (n t) d n_a')
            
            if self.cfg.double:
                nox, _ = online_model.backbone(next_obs)
                nox, _ = online_model.neck(nox, game_id)
                next_online_q_dist, _ = online_model.head(nox, game_id)
                next_online_q_dist = rearrange(next_online_q_dist, 'n t d n_a -> (n t) d n_a')                
                next_online_q =  (next_online_q_dist * self.support.reshape(1,1,-1)).sum(-1)
                next_act = torch.argmax(next_online_q, 1)
            else:       
                next_target_q =  (next_target_q_dist * self.support.reshape(1,1,-1)).sum(-1)     
                next_act = torch.argmax(next_target_q, 1)  
                
            next_act_idx = next_act.reshape(-1,1,1).repeat(1,1,self.num_atoms)
            target_q_dist = next_target_q_dist.gather(1, next_act_idx).squeeze(1)
        
            # C51 (https://arxiv.org/abs/1707.06887, Algorithm 1)
            # Compute the projection 
            # Tz = R_n + (γ^n)Z (w/ n-step return) (N, N_A)
            gamma_n = (gamma ** n_step)
            Tz = G.unsqueeze(-1) + gamma_n * self.support.unsqueeze(0) * (1-done).unsqueeze(-1)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            # L2-projection
            b = (Tz - self.v_min) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = torch.zeros((n, self.num_atoms), device=self.device)
            for idx in range(n):
                # += operation do not allow to add value to same index multiple times
                m[idx].index_add_(0, l[idx], target_q_dist[idx] * (u[idx] - b[idx]))
                m[idx].index_add_(0, u[idx], target_q_dist[idx] * (b[idx] - l[idx]))
                            
        # kl-divergence KL(p||q)=plogp-plogq
        # Here, plogp is just a constant
        EPS = 1e-5
        kl_div = -torch.sum(m * log_pred_q_dist, -1)
        kl_div = torch.clamp(kl_div, EPS, 1 / EPS)
        
        if reduction == 'mean':
            loss = (kl_div * weights).mean()
        else:
            loss = (kl_div * weights)

        # update priority
        if (self.buffer.name == 'per_buffer') and (mode == 'train'):
            self.buffer.update_priorities(idxs=tree_idxs, priorities=kl_div.detach().cpu().numpy())

        # prediction and target
        pred_q_dist = torch.exp(log_pred_q_dist)
        preds = (pred_q_dist * self.support.reshape(1,-1)).sum(-1)
        target_q = (target_q_dist * self.support.reshape(1,-1)).sum(-1)
        targets = G + gamma_n * target_q * (1-done)

        return loss, preds, targets        
