import numpy as np
import os
import re
import ale_py # atari_py
import cv2
import copy
from .base import *
from multiprocessing import Process, Pipe, Value
from src.common.class_utils import save__init__args
from gym.utils import seeding
from gym import spaces

"""
Modifies the default AtariEnv to be closer to DeepMind's setup.
Follows mila-iqia/spr's env for the most part.
"""

def get_game_path(game):
    import AutoROM
    basepath = os.path.dirname(AutoROM.__file__)
    gamepath = os.path.join(basepath, 'roms', game) + '.bin'
    return gamepath

class AtariEnv(BaseEnv):
    """
    An efficient implementation of the classic Atari RL envrionment using the
    Arcade Learning Environment (ALE).
    Output `env_info` includes:
        * `game_score`: raw game score, separate from reward clipping.
        * `traj_done`: special signal which signals game-over or timeout, so that sampler doesn't reset the environment when ``done==True`` but ``traj_done==False``, which can happen when ``episodic_lives==True``.
    Always performs 2-frame max to avoid flickering (this is pretty fast).
    The action space is an `IntBox` for the number of actions.  The observation
    space is an `IntBox` with ``dtype=uint8`` to save memory; conversion to float
    should happen inside the agent's model's ``forward()`` method.
    (See the file for implementation details.)
    Args:
        game (str): game name
        frame_skip (int): frames per step (>=1)
        frame (int): number of frames in observation (>=1)
        minimal_action_set (bool): whether to discard invalid actions, which depends on the game
        clip_reward (bool): if ``True``, clip reward to np.sign(reward)
        episodic_lives (bool): if ``True``, output ``done=True`` but ``env_info[traj_done]=False`` when a life is lost
        max_start_noops (int): upper limit for random number of noop actions after reset
        repeat_action_probability (0-1): probability for sticky actions
        horizon (int): max number of steps before timeout / ``traj_done=True``
    """
    name = 'atari'
    def __init__(self,
                 game="pong",
                 frame_skip=4,  # Frames per step (>=1).
                 frame=4,  # Number of (past) frames in observation (>=1).
                 minimal_action_set=True,
                 clip_reward=True,
                 episodic_lives=True,
                 max_start_noops=30,
                 repeat_action_probability=0.,
                 horizon=9000,  # 3000 for chopper command
                 stack_actions=0,
                 grayscale=True,
                 imagesize=84,
                 seed=42,
                 id=0,
                 ):
                 
        save__init__args(locals(), underscore=True)
        # ALE
        game = re.sub(r'(?<!^)(?=[A-Z])', '_', game).lower()
        game_path = get_game_path(game)
        if not os.path.exists(game_path):
            raise IOError(f"You asked for game {game} but path {game_path} does not exist")
            
        game_list = sorted(ATARI_RANDOM_SCORE.keys())
        if self._game in game_list:
            self._game_id = game_list.index(self._game)
        else:
            self._game_id = 0 # for ood games

        self.ale = ale_py.ALEInterface() # atari_py.ALEInterface()
        self.seed(seed, id)
        self.ale.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.ale.loadROM(game_path)

        # Spaces
        self.stack_actions = stack_actions
        self._minimal_action_set = minimal_action_set
        self._action_set = self.ale.getMinimalActionSet()
        if self._game in ['video_chess', 'tic_tac_toe_3d', 'miniature_golf', 'video_cube', \
                          'basic_math', 'othello', 'tetris', 'basic_math']:
            self._action_set = self._action_set[1:]
        self._action_space = spaces.Discrete(n=len(self._action_set))
        
        self.channels = 1 if grayscale else 3
        self.grayscale = grayscale
        self.imagesize = imagesize
        if self.stack_actions: self.channels += 1
        obs_shape = (frame, self.channels, imagesize, imagesize)
        self._observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )
        self._max_frame = self.ale.getScreenGrayscale() if self.grayscale \
            else self.ale.getScreenRGB()
        self._raw_frame_1 = self._max_frame.copy()
        self._raw_frame_2 = self._max_frame.copy()
        self._obs = np.zeros(shape=obs_shape, dtype="uint8")

        # Settings
        self._has_fire = "FIRE" in self.get_action_meanings()
        self._has_up = "UP" in self.get_action_meanings()
        self._horizon = int(horizon)
        self.reset()

    def seed(self, seed=None, id=0):
        _, seed1 = seeding.np_random(seed)
        if id > 0:
            seed = seed*100 + id
        self.np_random, _ = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)

    def reset(self):
        """Performs hard reset of ALE game."""
        self.ale.reset_game()
        self._reset_obs()
        self._life_reset()
        if self._max_start_noops > 0:
            for _ in range(self.np_random.randint(1, self._max_start_noops + 1)):
                self.ale.act(0)
                if self._check_life():
                    self.reset()
        self._update_obs(0)  # (don't bother to populate any frame history)
        self._step_counter = 0
        return self.get_obs()

    def step(self, action):
        if self._minimal_action_set:
            a = self._action_set[action]
        else:
            a = action
        game_score = np.array(0., dtype="float32")
        for _ in range(self._frame_skip - 1):
            game_score += self.ale.act(a)
        self._get_screen(1)
        game_score += self.ale.act(a)
        lost_life = self._check_life()  # Advances from lost_life state.
        if lost_life and self._episodic_lives:
            self._reset_obs()  # Internal reset.
        self._update_obs(action)
        reward = np.sign(game_score) if self._clip_reward else game_score
        game_over = self.ale.game_over() or self._step_counter >= self.horizon
        done = game_over or (self._episodic_lives and lost_life)
        info = EnvInfo(game_score=game_score, traj_done=game_over)
        self._step_counter += 1
        return EnvStep(self.get_obs(), reward, done, info)

    def render(self, wait=10, show_full_obs=False):
        """Shows game screen via cv2, with option to show all frames in observation."""
        img = self.get_obs()
        if show_full_obs:
            shape = img.shape
            img = img.reshape(shape[0] * shape[1], shape[2])
        else:
            img = img[-1]
        cv2.imshow(self._game, img)
        cv2.waitKey(wait)

    def get_obs(self):
        return self._obs.copy()

    ###########################################################################
    # Helpers

    def _get_screen(self, frame=1):
        frame = self._raw_frame_1 if frame == 1 else self._raw_frame_2
        if self.grayscale:
            self.ale.getScreenGrayscale(frame)
        else:
            self.ale.getScreenRGB(frame)

    def _update_obs(self, action):
        """Max of last two frames; crop two rows; downsample by 2x."""
        self._get_screen(2)
        np.maximum(self._raw_frame_1, self._raw_frame_2, self._max_frame)
        img = cv2.resize(self._max_frame, (self.imagesize, self.imagesize), cv2.INTER_LINEAR)
        if len(img.shape) == 2:
            img = img[np.newaxis]
        else:
            img = np.transpose(img, (2, 0, 1))
        if self.stack_actions:
            action = int(255.*action/self._action_space.n)
            action = np.ones_like(img[:1])*action
            img = np.concatenate([img, action], 0)
        # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
        self._obs = np.concatenate([self._obs[1:], img[np.newaxis]])

    def _reset_obs(self):
        self._obs[:] = 0
        self._max_frame[:] = 0
        self._raw_frame_1[:] = 0
        self._raw_frame_2[:] = 0

    def _check_life(self):
        lives = self.ale.lives()
        lost_life = (lives < self._lives) and (lives > 0)
        if lost_life:
            self._life_reset()
        return lost_life

    def _life_reset(self):
        self.ale.act(0)
        self._lives = self.ale.lives()

    ###########################################################################
    # Properties

    @property
    def game(self):
        return self._game
    
    @property
    def game_id(self):
        return self._game_id

    @property
    def action_size(self):
        return self._action_space.n

    @property
    def frame_skip(self):
        return self._frame_skip

    @property
    def frame(self):
        return self._frame

    @property
    def clip_reward(self):
        return self._clip_reward

    @property
    def max_start_noops(self):
        return self._max_start_noops

    @property
    def episodic_lives(self):
        return self._episodic_lives

    @property
    def repeat_action_probability(self):
        return self._repeat_action_probability

    @property
    def horizon(self):
        return self._horizon

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

ACTION_INDEX = {v: k for k, v in ACTION_MEANING.items()}

########################
# FAR-OOD game scores #
########################

FAR_OOD_RANDOM_SCORE = {
    'basic_math': 0.0033,
    'human_cannonball': 1.0767,
    'klax': 0.3734,
	'othello': -21.3233,
    'surround': -9.9833,
}

FAR_OOD_HUMAN_SCORE = {
    'basic_math': 1e8,
    'human_cannonball': 1e8,
    'klax': 1e8,
	'othello': 1e8,
    'surround': 1e8,
}

FAR_OOD_RAINBOW_SCORE = {
    'basic_math': 3.57,
    'human_cannonball': 5.56,
    'klax': 2124.75,
	'othello': -2.01,
    'surround': -7.81,
}

##############################
# ID & NEAR-OOD game scores #
##############################
# Game: score-tuple dictionary. Each score tuple contains
#  0: score random (float) and 1: score human (float).
# ref: https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/atari_data.py
# ref: Is Deep Reinforcement Learning Really Superhuman on Atari?
# since 5 games do not have human scores (air_raid, carnival, elevator_action, journey_escape, and, pooyan) 
# The human score is measured by playing games by myself. 

ATARI_RANDOM_SCORE = {
    'air_raid': 579.25,
    'alien': 227.8,
    'amidar': 5.8, 
    'assault': 222.4, 
    'asterix': 210.0, 
    'asteroids': 719.1, 
    'atlantis': 12850.0, 
    'bank_heist': 14.2, 
    'battle_zone': 2360.0, 
    'beam_rider': 363.9, 
    'berzerk': 123.7, 
    'bowling': 23.1, 
    'boxing': 0.1, 
    'breakout': 1.7, 
    'carnival': 700.8, 
    'centipede': 2090.9,
    'chopper_command': 811.0, 
    'crazy_climber': 10780.5, 
    'demon_attack': 152.1, 
    'double_dunk': -18.6, 
    'elevator_action': 4387.0, 
    'enduro': 0.0,
    'fishing_derby': -91.7, 
    'freeway': 0.0, 
    'frostbite': 65.2, 
    'gopher': 257.6, 
    'gravitar': 173.0, 
    'hero': 1027.0, 
    'ice_hockey': -11.2, 
    'jamesbond': 29.0, 
    'journey_escape': -19977.0, 
    'kangaroo': 52.0, 
    'krull': 1598.0, 
    'kung_fu_master': 258.5,
    'montezuma_revenge': 0.0, 
    'ms_pacman': 307.3, 
    'name_this_game': 2292.3, 
    'phoenix': 761.4, 
    'pitfall': -229.4, 
    'pong': -20.7, 
    'pooyan': 371.2, 
    'private_eye': 24.9,
    'qbert': 163.9, 
    'riverraid': 1338.5, 
    'road_runner': 11.5, 
    'robotank': 2.2, 
    'seaquest': 68.4, 
    'skiing': -17098.1, 
    'solaris': 1236.3, 
    'space_invaders': 148.0, 
    'star_gunner': 664.0, 
    'tennis': -23.8, 
    'time_pilot': 3568.0, 
    'tutankham': 11.4, 
    'up_n_down': 533.4, 
    'venture': 0.0, 
    # Note the random agent score on Video Pinball is sometimes greater than the
    # human score under other evaluation methods.
    'video_pinball': 16256.9, 
    'wizard_of_wor': 563.5, 
    'yars_revenge': 3092.9, 
    'zaxxon': 32.5, 
}

ATARI_HUMAN_SCORE = {
    'air_raid': 3325.0,
    'alien': 7127.7,
    'amidar': 1719.5,
    'assault': 742.0,
    'asterix': 8503.3,
    'asteroids': 47388.7,
    'atlantis': 29028.1,
    'bank_heist': 753.1,
    'battle_zone': 37187.5,
    'beam_rider': 16926.5,
    'berzerk': 2630.4,
    'bowling': 160.7,
    'boxing': 12.1,
    'breakout': 30.5,
    'carnival': 22870.0,
    'centipede': 12017.0,
    'chopper_command': 7387.8,
    'crazy_climber': 35829.4,
    'demon_attack': 1971.0,
    'double_dunk': -16.4,
    'elevator_action': 6750.5,
    'enduro': 860.5,
    'fishing_derby': -38.7,
    'freeway': 29.6,
    'frostbite': 4334.7,
    'gopher': 2412.5,
    'gravitar': 3351.4,
    'hero': 30826.4,
    'ice_hockey': 0.9,
    'jamesbond': 302.8,
    'journey_escape': 1129.3,
    'kangaroo': 3035.0,
    'krull': 2665.5,
    'kung_fu_master': 22736.3,
    'montezuma_revenge': 4753.3,
    'ms_pacman': 6951.6,
    'name_this_game': 8049.0,
    'phoenix': 7242.6,
    'pitfall': 6463.7,
    'pong': 14.6,
    'pooyan': 3004.3,
    'private_eye': 69571.3,
    'qbert': 13455.0,
    'riverraid': 17118.0,
    'road_runner': 7845.0,
    'robotank': 11.9,
    'seaquest': 42054.7,
    'skiing': -4336.9,
    'solaris': 12326.7,
    'space_invaders': 1668.7,
    'star_gunner': 10250.0,
    'tennis': -8.3,
    'time_pilot': 5229.2,
    'tutankham': 167.6,
    'up_n_down': 11693.2,
    'venture': 1187.5,
    'video_pinball': 17667.9,
    'wizard_of_wor': 4756.5,
    'yars_revenge': 54576.9,
    'zaxxon': 9173.3,
}

ATARI_DQN50M_SCORE = {
    'air_raid': 7479.50,
    'alien': 2484.49,
    'amidar': 1207.74,
    'assault': 1525.24,
    'asterix': 2711.41,
    'asteroids': 698.37,
    'atlantis': 853640.00,
    'bank_heist': 601.79,
    'battle_zone': 17784.84,
    'beam_rider': 5852.42,
    'berzerk': 487.48,
    'bowling': 30.12,
    'boxing': 77.99,
    'breakout': 96.23,
    'carnival': 4784.84,
    'centipede': 2583.03,
    'chopper_command': 2690.61,
    'crazy_climber': 104568.76,
    'demon_attack': 6361.58,
    'double_dunk': -6.54,
    'elevator_action': 439.77,
    'enduro': 628.91,
    'fishing_derby': 0.58,
    'freeway': 26.27,
    'frostbite': 367.07,
    'gopher': 5479.90,
    'gravitar': 330.07,
    'hero': 17325.44,
    'ice_hockey': -5.84,
    'jamesbond': 573.31,
    'journey_escape': -3671.09,
    'kangaroo': 11485.98,
    'krull': 6097.63,
    'kung_fu_master': 23435.38,
    'montezuma_revenge': 0.00,
    'ms_pacman': 3402.40,
    'name_this_game': 7278.65,
    'phoenix': 4996.58,
    'pitfall': -73.81,
    'pong': 16.61,
    'pooyan': 3211.96,
    'private_eye': -16.04,
    'qbert': 10117.50,
    'riverraid': 11638.93,
    'road_runner': 36925.47,
    'robotank': 59.77,
    'seaquest': 1600.66,
    'skiing': -15824.61,
    'solaris': 1436.40,
    'space_invaders': 1794.24,
    'star_gunner': 42165.22,
    'tennis': -1.50,
    'time_pilot': 3654.37,
    'tutankham': 103.84,
    'up_n_down': 8488.31,
    'venture': 39.13,
    'video_pinball': 63406.11,
    'wizard_of_wor': 2065.80,
    'yars_revenge': 23909.38,
    'zaxxon': 4538.57,
}