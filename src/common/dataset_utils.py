import gzip
import re
import os
import tqdm
import numpy as np
import torch
import h5py
import copy
from pathlib import Path
from typing import List, Tuple, Dict

FILE_TYPES = ['observation', 'action', 'reward', 'terminal', 'rtg', 'game_id']

def prepare_ataripb_dataset(replay_dataset_path: Path,
                            ataripb_dataset_path: Path,
                            ataripb_subdir_name: str,
                            games: List[str],
                            runs: List[int],
                            checkpoints: List[int],
                            per_ckpt_size: int,
                            n_step: int,
                            gamma: float,
                            minimal_action_set: bool
                            ) -> Dict[str, Path]:
    """
    Returns a dict of filepaths to Atari-PB dataset, while checking if the dataset exists.
    If there's no pre-processed Atari-PB dataset, create one.

    replay_dataset_path: Path to top directory of the dataset to sample from (e.g., DQN-Replay-Dataset).
    ataripb_dataset_path: Path to save/load preprocessed Atari-PB dataset.
    ataripb_subdir_name: Subdirectory name for each Atari-PB dataset (e.g., train, test, ...)
    games: List of all games to be included in the dataset.
    runs: List of run indexes for each game.
    checkpoints: List of checkpoint indexes for each run.
    per_ckpt_size: Number of transitions to be sampled from each checkpoint.
    n_step: n steps used by the algorithm.
    gamma: Discount factor.
    minimal_action_set: Whether the dataset used minimal action set.
    """

    file_paths = {}
    for file_type in FILE_TYPES:
        if file_type == 'observation': # obs takes the majority of space; share them among same per_ckpt_size.
            file_path = Path(f"{ataripb_dataset_path}/{ataripb_subdir_name}/{file_type}_{per_ckpt_size}.hdf5")
        else:
            file_path = Path(f"{ataripb_dataset_path}/{ataripb_subdir_name}/{file_type}_{per_ckpt_size}_{n_step}_{gamma}.hdf5")
        file_paths[file_type] = file_path
    
    # Check if dataset already exists, create one if it doesn't
    dataset_exists = all(os.path.isfile(file_path) for file_path in file_paths.values())
    obs_exists = os.path.isfile(file_paths['observation'])

    if not dataset_exists:
        create_ataripb_dataset(replay_dataset_path,
                               file_paths,
                               obs_exists,
                               games,
                               runs,
                               checkpoints,
                               per_ckpt_size,
                               n_step,
                               gamma,
                               minimal_action_set)
    
    return file_paths
    
def create_ataripb_dataset(replay_dataset_path: Path,
                           file_paths: Dict[str, Path],
                           obs_exists: bool,
                           games: List[str],
                           runs: List[int],
                           checkpoints: List[int],
                           per_ckpt_size: int,
                           n_step: int,
                           gamma: float,
                           minimal_action_set: bool
                           ) -> None:
    """
    Create Atari-PB dataset.

    replay_dataset_path: Path to top directory of the dataset to sample from (e.g., DQN-Replay-Dataset).
    file_paths: Files to individual data types (e.g., observation, action, ...)
    obs_exists: Whether observation is already pre-processed by other instances with same per_ckpt_size.
    games: List of all games to be included in the dataset.
    runs: List of run indexes for each game.
    checkpoints: List of checkpoint indexes for each run.
    per_ckpt_size: Number of transitions to be sampled from each checkpoint.
    n_step: n steps used by the algorithm.
    gamma: Discount factor.
    minimal_action_set: Whether the dataset used minimal action set.
    """

    # Create local hdf5 files.
    # hdf5 file will be created for each obs, action, etc.
    # Each hdf5 file will contain one dataset named 'data'.
    file_dict = {}
    for file_type, file_path in file_paths.items():
        if file_type == 'observation' and obs_exists:
            continue
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_dict[file_type] = h5py.File(file_path, 'w')

    # Gather dataset from replay_dataset_path
    data_dict = {}
    for game_idx, game in enumerate(games):
        for run_idx, run in enumerate(runs):
            for ckpt_idx, ckpt in enumerate(checkpoints):
                dataset = load_replay_dataset(replay_dataset_path,
                                              obs_exists,
                                              game,
                                              run,
                                              ckpt,
                                              per_ckpt_size,
                                              n_step,
                                              gamma,
                                              minimal_action_set)

                for file_type, data_ in dataset.items():
                    if 'data' not in file_dict[file_type]:
                        n_games, n_runs, n_ckpts = len(games), len(runs), len(checkpoints)
                        data_dict[file_type] = file_dict[file_type].create_dataset("data",(n_games, n_runs, n_ckpts, *data_.shape), dtype=data_.dtype)
                    data_dict[file_type][game_idx,run_idx,ckpt_idx,...] = copy.deepcopy(data_)

                del dataset

    print("Dataset successfully gathered and saved in hdf5!")
    
    del data_dict
    for f in file_dict.values():
        f.close()

def load_replay_dataset(data_path: Path,
                        obs_exists: bool,
                        game: str,
                        run: int,
                        checkpoint: int,
                        num_samples: int,
                        n_step: int,
                        gamma: float,
                        minimal_action_set: bool
                        ) -> Dict[str, np.ndarray]:
    """
    Load a certain checkpoint in the replay dataset and preprocess it.

    data_path: Path to top directory of the dataset to sample from (e.g., DQN-Replay-Dataset).
    obs_exists: Whether observation is already pre-processed by other instances with same num_samples.
    game: Name of game to sample from.
    run: Index of run to sample from.
    checkpoint: Index of checkpoint to sample from.
    num_samples: Number of transitions to be sampled from each checkpoint.
    n_step: n steps used by the algorithm.
    gamma: Discount factor.
    minimal_action_set: Whether the dataset used minimal action set.
    """

    dataset = {} 
    _game = ''.join(word.capitalize() for word in str(game).split('_'))

    print(f"Loading from {data_path}: {game}, run {run}, checkpoint {checkpoint}")

    def load_from_gz(filepath):
        g = gzip.GzipFile(filename=filepath)
        data__ = np.load(g)
        data_ = np.copy(data__[:num_samples])
        print(f'Using {data_.size * data_.itemsize} bytes')
        del data__
        return data_

    for filetype in FILE_TYPES:
        gz_filepath = Path(data_path + f'/{_game}/{filetype}_{run}_{checkpoint}.gz')

        # Observation    
        if filetype == 'observation':

            # 1. Observation is shared among those that use the same num_samples.
            if obs_exists:
                print("Observation has already been preprocessed. Skipping.")
                continue
            
            # 2. Extracting from gz file is slow. This is unavoidable in the first run,
            # but if the dataset has to be created in multiple machines locally,
            # creating a npy version of the subset will optimize the speed afterwards.
            npy_filepath = Path(data_path + f'/{_game}/{filetype}_{run}_{checkpoint}_{num_samples}.npy')
            if not os.path.exists(npy_filepath):
                print("No .npy version of observation found. Creating one from .gz")
                data___ = load_from_gz(gz_filepath)
                np.save(npy_filepath, data___)
                print(f"Stored on disk at {npy_filepath}")
                del data___

            # Load from .npy
            with open(npy_filepath, "rb") as f:
                data_ = np.load(f)

            # unsqueeze channel dim if needed
            if len(data_.shape) == 3:
                data_ = np.expand_dims(data_, axis=1)
            
        # Action, Reward, Terminal - load directly from gz
        elif filetype == 'action':
            data_ = load_from_gz(gz_filepath)
            # Convert action set if needed
            if not minimal_action_set:
                action_mapping = dict(zip(np.unique(data_), AtariEnv(_game).ale.getMinimalActionSet()))
                data_.apply_(lambda x: action_mapping[x])
        elif filetype == 'reward':
            data_ = load_from_gz(gz_filepath)
        elif filetype == 'terminal':
            data_ = load_from_gz(gz_filepath)
            # Spread dones forward by n_steps
            for _ in range(n_step-1):
                nth_step_done = np.pad(data_[1:],(0,1)) >= 1                     
                data_ = data_ + nth_step_done
            data_ = data_ >= 1
            
        # RTG - Requires Reward and Terminal to be processed first.
        elif filetype == 'rtg':
            npy_filepath = Path(data_path + f'/{_game}/{filetype}_{run}_{checkpoint}_{num_samples}_{n_step}_{gamma}.npy')
            if not os.path.exists(npy_filepath): 
                print(f'Pre-processed rtg not found. Creating one.')
                print(f'Compute rtg with n_step:%d, gamma:%.2f' %(n_step, gamma))

                # Get previously processed reward/terminal
                rewards = np.sign(np.nan_to_num(dataset['reward'])) # nan reward exists for some reason
                dones = dataset['terminal']
                
                # compute rtg based on whole trajectory
                # this config is used in decision transformer
                # https://arxiv.org/abs/2106.01345
                if (n_step == 0) and (gamma == 1.0):
                    # compute returns for each trajectory
                    G = 0
                    traj_start_idx = 0
                    return_per_trajectory = []
                    for idx in tqdm.tqdm(range(len(rewards))):
                        reward = rewards[idx].item()
                        terminal = dones[idx].item()
                        
                        G += reward
                        if terminal == 1:
                            return_per_trajectory.append(G)
                            G = 0
                            traj_start_idx = idx+1
                
                    # last trajectory
                    return_per_trajectory.append(G)    
                        
                    # compute rtg for each interaction    
                    rtgs = np.zeros_like(rewards)
                    traj_idx = 0
                    G = return_per_trajectory[traj_idx]
                    for idx in tqdm.tqdm(range(len(rewards))):
                        reward = rewards[idx].item()
                        terminal = dones[idx].item()
                        
                        rtgs[idx] = G
                        G -= reward
                        
                        if terminal == 1:
                            traj_idx += 1
                            G = return_per_trajectory[traj_idx]
                
                # compute rtg given n_step and gamma   
                else:
                    rtgs = np.zeros_like(rewards)
                    for step in reversed(range(n_step)):
                        n_step_reward = np.concatenate((rewards[step:], np.zeros(step)))
                        n_step_done = np.concatenate((dones[step:], np.zeros(step)))                            
                        rtgs = n_step_reward + gamma * rtgs * (1 - n_step_done)

                data__ = rtgs                    
                np.save(npy_filepath, data__,)   
                print("Stored on disk at {}".format(npy_filepath))
                del data__
            
            
            data_ = np.load(npy_filepath)
            if len(data_) != num_samples:
                raise ValueError("This error exists for our internal backward compatibiity. If this is raised, please contact us.")
        
        # Game ID
        elif filetype == 'game_id':
            game_list = sorted(PRETRAIN_GAME_LIST)
            if game in game_list:
                game_id = game_list.index(game)
            else:
                game_id = 0
            data_ = np.array([game_id]*num_samples)

        else:
            raise ValueError
    
        dataset[filetype] = data_

    return dataset

# DO NOT MODIFY THIS LIST
PRETRAIN_GAME_LIST = ['air_raid', 'alien', 'amidar', 'assault', 'asterix', \
                      'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 'beam_rider', \
                      'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', \
                      'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk', \
                      'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', \
                      'gopher', 'gravitar', 'hero', 'ice_hockey', 'jamesbond', \
                      'journey_escape', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge', \
                      'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', \
                      'pooyan', 'private_eye', 'qbert', 'riverraid', 'road_runner', \
                      'robotank', 'seaquest', 'skiing', 'solaris', 'space_invaders', \
                      'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', \
                      'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']