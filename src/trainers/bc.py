import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from einops import rearrange
from .base import BaseTrainer
from src.common.logger import AgentLogger


class BCTrainer(BaseTrainer):
    name = 'bc'
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
        ##############
        # forward
        x = obs = batch['obs']        
        act = batch['act']
        game_id = batch['game_id']
        
        # augmentation
        n, t, f, c, h, w = x.shape
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        # online encoder
        x, _ = self.model.backbone(x)
        x, _ = self.model.neck(x, game_id) # game-wise spatial embedding
        act_pred, _ = self.model.head(x, game_id) # game-wise prediction head
                
        # loss
        loss_fn = nn.CrossEntropyLoss()
        act_pred = rearrange(act_pred, 'n t d -> (n t) d')
        act = rearrange(act, 'n t -> (n t)')
        
        loss = loss_fn(act_pred, act)
        act_acc = torch.mean((torch.argmax(act_pred, 1) == act).float())

        ###############
        # logs        
        log_data = {'loss': loss.item(),
                    'act_acc': act_acc.item()}
        
        return loss, log_data
    
    def update(self, batch, step):
        raise NotImplementedError
    
    def rollout(self):
        # rollout is only performed when
        # (1) evaluation environment is vectorized 
        # (2) a single game is vectorized
        # (3) number of environments == number of evaluations
        game = self.eval_env.game[0] 
        eval_env_action_size = self.eval_env.action_size[0] 
        
        game_id = self.eval_env.game_id # (n, )
        game_id = torch.LongTensor(game_id).to(self.device)
        game_id = rearrange(game_id, 'n -> n 1')       
        
        agent_logger = AgentLogger(
            num_envs = self.cfg.num_eval_envs,
            game = game
        )
        obs = self.eval_env.reset() # (n, 4, 1, 84, 84)
        while True:
            with torch.no_grad():
                # encode obs
                x = torch.FloatTensor(obs).to(self.device)
                x = x / 255.0
                
                # get arg-max action
                x = rearrange(x, 'n f c h w -> n 1 f c h w')
                x, _ = self.model.backbone(x)
                x, _ = self.model.neck(x, game_id) # game-wise spatial embedding
                act_pred, _ = self.model.head(x, game_id) # game-wise prediction head
                act_pred = rearrange(act_pred, 'n t d -> (n t) d')
            
            # mask the non-minimal actions
            # In training, the action size is defined as the maximum number of actions
            # In evaluation, the action should be limited to the available action in each env.
            INF = 1e8
            act_pred[:, eval_env_action_size:] = -INF
            argmax_action = torch.argmax(act_pred, 1)
            argmax_action = argmax_action.cpu().numpy()
            
            # eps-greedy
            eps = self.cfg.eval_eps
            prob = np.random.rand(self.cfg.num_eval_envs)
            is_rand = (prob <= eps)
            rand_action = np.random.randint(0, eval_env_action_size-1, self.cfg.num_eval_envs)
            action = is_rand * rand_action + (1-is_rand) * argmax_action

            # step
            next_obs, reward, done, info = self.eval_env.step(action)
            
            # logger
            agent_logger.step(obs, reward, done, info)
            
            # move on
            if agent_logger.is_traj_done:
                break
            else:
                obs = next_obs
    
        log_data = agent_logger.fetch_log('eval')
        
        return log_data
