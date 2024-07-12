import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import tqdm
from collections import defaultdict, deque
from einops import rearrange
from .base import BaseTrainer
from src.common.logger import AgentLogger


class DTTrainer(BaseTrainer):
    name = 'dt'
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
            
    def _compute_max_rtg(self):
        max_rtg_dict = defaultdict(lambda: -1e8, {})

        for batch in tqdm.tqdm(self.train_loader):   
            rtg = batch['rtg'].flatten().numpy()
            game_id = batch['game_id'].flatten().numpy()
            
            for _rtg, _game_id in zip(rtg, game_id):
                if _rtg > max_rtg_dict[_game_id]:
                    max_rtg_dict[_game_id] = _rtg
        
        return dict(max_rtg_dict)
    
    def train(self):
        self.max_rtg_dict = self._compute_max_rtg()
        super().train()
            
    def compute_loss(self, batch):
        ##############
        # forward
        x = obs = batch['obs']    
        act = batch['act']
        done = batch['done'].type(torch.int64)
        rtg = batch['rtg'] * self.cfg.rtg_scale # ensure to have a small value
        game_id = batch['game_id']
        
        # augmentation
        n, t, f, c, h, w = x.shape
        x = rearrange(x, 'n t f c h w -> n (t f c) h w')
        x = self.aug_func(x)
        x = rearrange(x, 'n (t f c) h w -> n t f c h w', t=t, f=f, c=c)

        # online encoder
        x, _ = self.model.backbone(x)
        x, _ = self.model.neck(x, act, rtg, game_id) # game-wise spatial embedding
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
        
        #####################################################
        # 1. initialize history for decision transformer
        n, t = self.cfg.num_eval_envs, self.cfg.t_step
        
        obs_hist = torch.zeros((n, t, *self.cfg.obs_shape)).to(self.device)
        rtg_hist = torch.zeros((n, t)).to(self.device)
        act_hist = torch.zeros((n, t)).long().to(self.device)
                
        # get max rtg that was used in the training dataset
        self.max_rtg_dict = self.load_rtg_dict()
        for idx, g_id in enumerate(game_id):
            max_rtg = self.max_rtg_dict[g_id.item()]
            rtg_hist[idx][-1] = torch.FloatTensor([max_rtg]).to(self.device)
        game_id = game_id.repeat(1, t)
            
        ######################################################
        # 2. rollout
        # initilaize observation
        obs = self.eval_env.reset() # (n, 4, 1, 84, 84)
        obs_hist[:, -1] = torch.FloatTensor(obs).to(self.device)

        while True:
            with torch.no_grad():       
                obs_in = obs_hist / 255.0
                act_in = act_hist
                rtg_in = rtg_hist * self.cfg.rtg_scale

                x, _ = self.model.backbone(obs_in)
                x, _ = self.model.neck(x, act_in, rtg_in, game_id) # game-wise spatial embedding
                act_pred, _ = self.model.head(x, game_id) # game-wise prediction head
                act_pred = act_pred[:, -1]
        
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

                # update history
                obs_hist[:, :-1] = obs_hist[:, 1:]
                act_hist[:, :-1] = act_hist[:, 1:]
                rtg_hist[:, :-1] = rtg_hist[:, 1:]
                
                obs_hist[:, -1] = torch.FloatTensor(obs).to(self.device)
                act_hist[:, -1] = torch.LongTensor(action).to(self.device)
                rtg_hist[:, -1] -= torch.FloatTensor(reward).to(self.device)
    
        log_data = agent_logger.fetch_log('eval')
        
        return log_data
    
    def save_checkpoint(self, epoch, step, name=None):
        # Save only once
        if self.logger.rank == 0:
            if name is None:
                name = 'epoch'+str(epoch)
            save_dict = {
                'backbone': self.model.backbone.state_dict(),
                'neck': self.model.neck.state_dict(),
                'head': self.model.head.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'max_rtg': self.max_rtg_dict,
                'epoch': epoch,
                'step': step
            }
            
            self.logger.save_dict(save_dict=save_dict,
                                  name=name)

    def load_checkpoint(self):
        # Dict of elements to load
        load_dict = {
            'backbone': self.model.backbone,
            'neck': self.model.neck,
            'head': self.model.head,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'scaler': self.scaler,
            'max_rtg': self.max_rtg_dict,
            'epoch': -1,
            'step': -1
        }

        ret = self.logger.load_dict(load_dict=load_dict,
                                    path=self.cfg.ckpt_path,
                                    device=self.device)
        start_epoch = ret['epoch']+1
        start_step = ret['step']+1

        print('Starting at epoch', start_epoch)
        return start_epoch, start_step
    
    def load_rtg_dict(self):
        load_dict = {
            'max_rtg': {},
        }
        
        ret = self.logger.load_dict(
            load_dict=load_dict,
            path=self.cfg.ckpt_path,
            device=self.device
        )
        
        return ret['max_rtg']
