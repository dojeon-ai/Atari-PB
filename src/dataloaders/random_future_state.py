import numpy as np
import torch
from typing import List, Tuple
from torch.utils.data import Dataset
from .base import BaseDataset
from random import randint

class RandomFutureState(BaseDataset):
    name = 'random_future_state'
    def __init__(self, 
                 file_paths,
                 games: List[str],
                 runs: List[int],
                 checkpoints: List[int],
                 samples_per_checkpoint: int,
                 frame: int,
                 t_step: int,
                 n_step: int,
                 gamma: float,
                 **kwargs):

        super().__init__(file_paths=file_paths,
                        games=games,
                        runs=runs,
                        checkpoints=checkpoints,
                        samples_per_checkpoint=samples_per_checkpoint,
                        frame=frame,
                        t_step=t_step,
                        n_step=n_step,
                        gamma=gamma,
                        )

    def __getitem__(self, index: int):                  
        
        game_idx, run_idx, ckpt_idx, index = self.get_indexes(index)
        
        start_idx = index
        end_idx = start_idx + self.t_step + (self.frame-1)
        
        sl = slice(start_idx, end_idx)
        n_step = randint(1,self.n_step) # next state is any step in [t+1,t+n]
        nsl = slice(start_idx + n_step, end_idx + n_step)
        
        slice_dict = {file_type:sl for file_type in self.file_paths.keys()}
        slice_dict['next_observation'] = nsl

        item_dict = {}
        for file_type, _slice in slice_dict.items():
            item_dict[file_type] = self.access_file(file_type, game_idx, run_idx, ckpt_idx, _slice)

        return item_dict

