import torch
import torch.nn as nn
import numpy as np
from collections import deque
from .base import BaseBuffer
from einops import rearrange
import time
import os
import gzip


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.max = 1  # Initial max value to return (1 = 1^ω), default transition priority is set to max

     # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

     # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, value):
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    def total(self):
        return self.sum_tree[0]


class PERBuffer(BaseBuffer):
    name = 'per_buffer'
    def __init__(self, obs_shape, action_size, size, prior_exp, max_n_step, device):
        super().__init__()
        self.size = size
        self.prior_exp = prior_exp
        self.max_n_step = max_n_step
        self.device = device
        
        self.n_step_transitions = deque(maxlen=max_n_step)
        self.obs_buffer = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.act_buffer = np.zeros(size)
        self.rew_buffer = np.zeros(size)
        self.done_buffer = np.zeros(size)
        self.segment_tree = SegmentTree(size)
        self.num_in_buffer = 0        
        self.buffer_idx = 0
        self.ckpt = 1

    def store(self, obs, action, reward, done):
        self.n_step_transitions.append((obs, action, reward, done))
        if len(self.n_step_transitions) < self.max_n_step:
            return
        obs, action, reward, done = self.n_step_transitions[0]
        
        b_idx = self.buffer_idx
        self.obs_buffer[b_idx] = obs
        self.act_buffer[b_idx] = action
        self.rew_buffer[b_idx] = reward
        self.done_buffer[b_idx] = done
        
        # store new transition with maximum priority
        self.segment_tree.append(value=self.segment_tree.max)
        
        # increase buffer count
        self.num_in_buffer = min(self.num_in_buffer+1, self.size)
        b_idx = b_idx + 1
        if b_idx == self.size:
            b_idx = 0
        self.buffer_idx = b_idx

    # Returns a valid sample from each segment
    def _get_idxs_from_segments(self, batch_size):
        p_total = self.segment_tree.total() # sum of the priorities
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        
        while not valid:
            samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts  # Uniformly sample from within all segments
            probs, data_idxs, tree_idxs = self.segment_tree.find(samples)  # Retrieve samples from tree with un-normalised probability
            
            # extra conservative around buffer index 0
            # n_step must be stacked before sampling
            if np.all(probs != 0) and np.all(data_idxs < (self.size- self.max_n_step)):
                valid = True 
                
        return data_idxs, tree_idxs, probs

    def sample(self, batch_size, n_step=10, gamma=0.99, prior_weight=1.0):
        if self.num_in_buffer < batch_size:
            assert('Replay buffer does not have enough transitions to sample')
        data_idxs, tree_idxs, probs = self._get_idxs_from_segments(batch_size)

        # compute n-step returns
        n_step_idxs = data_idxs[:, np.newaxis] + np.arange(n_step)
        n_step_rew_batch = self.rew_buffer[n_step_idxs] # (batch_size, n_step)
        n_step_done_batch = self.done_buffer[n_step_idxs] # (batch_size, n_step)
        G = n_step_rew_batch[:, -1]
        done = n_step_done_batch[:, -1]
        next_obs_offset = np.ones(batch_size) * n_step
        for step in reversed(range(n_step-1)):
            rew = n_step_rew_batch[:, step]
            _done = n_step_done_batch[:, step]
            _next_obs_offset = np.ones(batch_size) * (step + 1)
            
            G = rew + gamma * G * (1-_done)
            next_obs_offset = (1-_done) * next_obs_offset + _done * _next_obs_offset
            done = np.logical_or(done, _done).astype('float')
        
        # get transitions
        obs_batch = self.obs_buffer[data_idxs]
        act_batch = self.act_buffer[data_idxs]
        rew_batch = rew
        done_batch = done
        G_batch = G
        next_obs_batch = self.obs_buffer[data_idxs + next_obs_offset.astype(int)]
        
        # convert to tensors
        obs_batch = self.encode_obs(obs_batch)
        act_batch = torch.LongTensor(act_batch).to(self.device)
        rew_batch = torch.FloatTensor(rew_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        G_batch = torch.FloatTensor(G_batch).to(self.device)
        next_obs_batch = self.encode_obs(next_obs_batch)
        
        # compute importance weights
        p_total = self.segment_tree.total()
        N = self.num_in_buffer
        probs = probs / p_total
        weights = (1 / (probs * N) + 1e-5) ** prior_weight
        
        # re-normalise by max weight (make update scale consistent w.r.t learning rate)
        weights = weights / max(weights)
        weights = torch.FloatTensor(weights).to(self.device)

        batch = {
            'obs': obs_batch,
            'act': act_batch,
            'rew': rew_batch,
            'done': done_batch,
            'G': G_batch,
            'next_obs': next_obs_batch,
            'n_step': n_step,
            'gamma': gamma,
            'tree_idxs': tree_idxs,
            'weights': weights,
        }
        return batch

    def encode_obs(self, obs, prediction=False):
        obs = np.array(obs, dtype=np.float32)
        obs = obs / 255.0

        # prediction: batch-size: 1
        if prediction:
            obs = np.expand_dims(obs, 0)
        
        obs = np.expand_dims(obs, 1)
        obs = torch.FloatTensor(obs).to(self.device)

        return obs

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.prior_exp)
        self.segment_tree.update(idxs, priorities)
        
    def save_buffer(self, buffer_dir, game, run):
        dir = buffer_dir + '/' + game + '/' 
        ckpt = self.ckpt
        observation_path = dir + 'observation' + '_' + str(run) + '_' + str(ckpt) + '.gz'
        action_path = dir + 'action' + '_' + str(run) + '_' + str(ckpt) + '.gz'
        reward_path = dir + 'reward' + '_' + str(run) + '_' + str(ckpt) + '.gz'
        terminal_path = dir + 'terminal' + '_' + str(run) + '_' + str(ckpt) + '.gz'        
        
        os.makedirs(dir, exist_ok=True)
        
        with gzip.open(observation_path, 'wb') as f:
            np.save(f, self.obs_buffer[:, -1]) # (n, c, h, w)
            
        with gzip.open(action_path, 'wb') as f:
            np.save(f, self.act_buffer) # (n,)

        with gzip.open(reward_path, 'wb') as f:
            np.save(f, self.rew_buffer) # (n,)
            
        with gzip.open(terminal_path, 'wb') as f:
            np.save(f, self.done_buffer) # (n,)
            
        self.ckpt += 1
