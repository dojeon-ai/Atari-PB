import copy
from omegaconf import OmegaConf
from src.common.dataset_utils import prepare_ataripb_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .base import BaseDataset
from src.common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseDataset)
DATASETS = {subclass.get_name():subclass
            for subclass in all_subclasses(BaseDataset)}

def build_dataloader(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Create eval cfg
    eval_ratio = cfg.pop('eval_ratio')
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg['samples_per_checkpoint'] = int(eval_ratio * cfg['samples_per_checkpoint'])
    eval_cfg['distributed'] = False
    eval_cfg['shuffle'] = False

    # Prepare AtariPB dataset and get filepaths
    file_paths = prepare_ataripb_dataset(
        replay_dataset_path=cfg['replay_dataset_path'],
        ataripb_dataset_path=cfg['ataripb_dataset_path'],
        ataripb_subdir_name=cfg['ataripb_subdir_name'],
        games=cfg['games'],
        runs=cfg['runs'],
        checkpoints=cfg['checkpoints'],
        per_ckpt_size=cfg['samples_per_checkpoint'],
        n_step=cfg['n_step'],
        gamma=cfg['gamma'],
        minimal_action_set=cfg['minimal_action_set']
    )

    train_dataset = get_dataset(cfg, file_paths)
    eval_dataset = get_dataset(eval_cfg, file_paths)
    train_dataloader, train_sampler = create_dataloader(cfg, train_dataset)
    eval_dataloader, eval_sampler = create_dataloader(eval_cfg, eval_dataset)
    
    return train_dataloader, train_sampler, eval_dataloader, eval_sampler

def get_dataset(cfg, file_paths):
    dataset_type = cfg['dataset_type']
    dataset_cls = DATASETS[dataset_type]
    dataset = dataset_cls(file_paths=file_paths, **cfg)
    return dataset

def create_dataloader(cfg, dataset):
    
    prefetch_factor = cfg['prefetch_factor']
    if cfg['num_workers'] == 0:
        prefetch_factor = None
    
    if cfg['distributed']:
        sampler = DistributedSampler(dataset, shuffle=cfg['shuffle'])
        shuffle = False
    else:
        sampler = None
        shuffle = cfg['shuffle']

    dataloader = DataLoader(dataset,
                            batch_size=cfg['batch_size'],
                            num_workers=cfg['num_workers'],
                            pin_memory=cfg['pin_memory'],
                            shuffle=shuffle,
                            sampler=sampler,
                            drop_last=False,
                            prefetch_factor=prefetch_factor
                           )
                           
    return dataloader, sampler