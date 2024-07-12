import numpy as np
import torch
import h5py
from os.path import splitext
from typing import List, Tuple
from torch.utils.data import Dataset
from src.common.class_utils import save__init__args

class BaseDataset(Dataset):
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
                 ):

        save__init__args(locals())
        self.ngames = len(games)
        self.nruns = len(runs)
        self.nckpts = len(checkpoints)
        self.effective_size = samples_per_checkpoint - (frame - 1) - (t_step - 1) - n_step
        self.file_paths['next_observation'] = self.file_paths['observation']
        _, self.file_suffix = splitext(self.file_paths['observation'])
        self.dataset_dict = None # For hdf5, see `load_file()`.
        
        # TODO: remove the below code later (temporarily utilized to use num_gpus: 3)
        if self.effective_size % 3 == 0:
            self.effective_size = self.effective_size
        else:
            self.effective_size = self.effective_size - (self.effective_size % 3)

    @classmethod
    def get_name(cls):
        return cls.name
        
    def __len__(self) -> int:
        return self.ngames * self.nruns * self.nckpts * self.effective_size

    def get_indexes(self, index):
        game_idx = index // (self.nruns * self.nckpts * self.effective_size)
        index %= self.nruns * self.nckpts * self.effective_size
        
        run_idx = index // (self.nckpts * self.effective_size)
        index %= self.nckpts * self.effective_size

        ckpt_idx = index // self.effective_size
        index %= self.effective_size

        return (game_idx, run_idx, ckpt_idx, index)

    def load_hdf5(self):
        # Called only once, in access_file()
        self.dataset_dict = {}
        for file_type, file_path in self.file_paths.items():
            f = h5py.File(file_path, 'r')
            self.dataset_dict[file_type] = f['data']

    def access_file(self, file_type, game_idx, run_idx, ckpt_idx, _slice):
        # Input: Desired file_type(obs,act,...) and indexes
        # Output: Tensor of the data of that type and index. Works differently according to file suffix.

        if self.file_suffix == '.npy':
            data_ = np.load(self.file_paths[file_type], mmap_mode='r')
        elif self.file_suffix == '.hdf5':
            if self.dataset_dict is None: # hdf5 works faster if it's loaded while __getitem__ is running.
                self.load_hdf5()
            data_ = self.dataset_dict[file_type]
        else:
            raise NotImplemented

        item = data_[game_idx,run_idx,ckpt_idx,_slice]
        item = torch.tensor(item)
        del data_

        return item 

    def __getitem__(self, index: int):                  
        pass
        
        ##################################################
        #  Considerations when implementing __getitem__  #
        ##################################################

        # 1. Use either numpy memmap or hdf5. 
        #     Although our numpy array is a large dataset (~100gb),
        #     np.load with memmap allows to only load the address which does not increase the computation cost.
        #
        # 2. When using memmap, np.load must be inside the __getitem__ function (and not __init__).
        #    Reason being, DDP with num_workers > 0 does not allow to pass np.array to its child.
        #     DDP: False, num_workers: 0 -> o
        #     DDP: False, num_workers> 0 -> o
        #     DDP: True,  num_workers: 0 -> o
        #     DDP: True,  num_workers> 0 -> x
        #    The official pytorch code of ImageNet is also implemented similarly
        #    where each image is loaded inside the __getitem__.
        

