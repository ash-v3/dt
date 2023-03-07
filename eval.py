import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
from dt import DecisionTransformer
from proc import StateProc
from replay_buffer import ReplayBuffer

EPOCHS = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("CarRacing-v2")
obs_dim = env.observation_space.shape
obs_dim = [obs_dim[2], obs_dim[0], obs_dim[1]]
act_dim = env.action_space.shape[0]
in_dim = 772

model = DecisionTransformer(in_dim=in_dim, n_heads=4)
state_proc = StateProc()
replay_buffer = ReplayBuffer()

observation, info = env.reset()

for e in range(EPOCHS):
    observation = env.reset()
    action = torch.zeros(act_dim, dtype=torch.float32)
    reward = torch.tensor([0], dtype=torch.float32)
    obs = torch.tensor(observation[0], dtype=torch.float32).reshape(obs_dim)
    obs = state_proc(obs, action, torch.tensor([reward]))
    print(obs.shape)
    hist = obs

    i = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = model(hist).detach().numpy()[0][-act_dim:] # later make get action instead of just last
        observation, reward, terminated, truncated, info = env.step(action)
        obs = torch.tensor(observation, dtype=torch.float32).reshape(obs_dim)
        obs = state_proc(obs, torch.tensor(action), torch.tensor([reward]))
        hist = torch.cat([hist, obs])
        i += 1
        if i % 10 == 0:
            terminated = True
        print(hist.shape, action.shape, obs.shape, replay_buffer.buffer.shape)

    replay_buffer.cat(hist)
