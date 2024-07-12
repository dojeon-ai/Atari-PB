import numpy as np
import torch
from typing import List, Tuple
from torch.utils.data import Dataset
from .base import BaseDataset

class TwoFutureStates(BaseDataset):
    name = 'two_future_states'
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
        self.next_steps = [kwargs['pos_next'], kwargs['neg_next']]

    def __getitem__(self, index):                  
        
        game_idx, run_idx, ckpt_idx, index = self.get_indexes(index)
        start_idx = index
        end_idx = start_idx + self.t_step + (self.frame-1)

        # list of slices
        sl = slice(start_idx, end_idx)
        nsl = [slice(start_idx + n_step, end_idx + n_step) for n_step in self.next_steps]

        # specify which slice list to use for each file type
        slice_dict = {file_type:sl for file_type in self.file_paths.keys()}
        slice_dict['next_observation'] = nsl
        
        # load data
        item_dict = {}
        for file_type, _slice in slice_dict.items():

            if file_type == 'next_observation':
                items = []
                for s in slice_dict[file_type]:
                    items.append(self.access_file(file_type, game_idx, run_idx, ckpt_idx, s))
                item = np.concatenate(items, axis=1)
            else:
                item = self.access_file(file_type, game_idx, run_idx, ckpt_idx, _slice)

            item_dict[file_type] = item

        return item_dict

