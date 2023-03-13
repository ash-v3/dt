import os
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
from datetime import datetime
from dt import DecisionTransformer

TARGET_RETURN = 10000
EPOCHS = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("CarRacing-v2")
state_dim = env.observation_space.shape
image_dim = [state_dim[2], state_dim[0], state_dim[1]]
act_dim = env.action_space.shape[0]
encoding_dim = 768

model = DecisionTransformer(state_dim=encoding_dim, act_dim=act_dim, device=device)

observation, info = env.reset()

now = datetime.now()
if not os.path.exists("./replays"):
    os.mkdir("./replays")
replay_folder = "./replays/" + now.strftime("%m_%d_%Y_%H_%M_%S")
os.mkdir(replay_folder)

for e in range(EPOCHS):
    state, _ = env.reset()
    actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
    encoding = model.proc_state(observation)
    states = encoding.reshape(1, 1, encoding_dim).to(device=device, dtype=torch.float32)
    target_returns = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)
    with torch.inference_mode():
        state_preds, action_preds, return_preds = model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=target_returns,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

    terminated = truncated = False
    while not (terminated or truncated):
        with torch.inference_mode():
            # print(target_returns.shape, states.shape, actions.shape)

            state_preds, action_preds, return_preds = model(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=target_returns,
                timesteps=timesteps,
                attention_mask=attention_mask,
                return_dict=False,
            )

            action = action_preds[0, -1].cpu().numpy()

        observation, reward, terminated, truncated, info = env.step(action)

        # save proper stuff, backwards update RTG, format right
        timesteps[0, 0] += 1
        actions = torch.cat([actions, torch.from_numpy(action).to(device=device).reshape([1, 1, act_dim])], dim=1).to(device=device)
        rewards = torch.cat([rewards, torch.tensor([reward], device=device).unsqueeze(0)], dim=0).to(device=device)
        encoding = model.proc_state(observation).to(device=device)
        target_returns = torch.cat([target_returns, return_preds[0, -1].reshape([1, 1, 1]).to(device=device)], dim=1).to(device=device)
        states = torch.cat([states, encoding.reshape([1, 1, encoding_dim])], dim=1).to(device=device)
        attention_mask = torch.cat([attention_mask, torch.tensor(0, device=device).reshape([1, 1])], dim=1).to(device=device)

    torch.save(states, replay_folder + "/states_" + str(e))
    torch.save(actions, replay_folder + "/actions_" + str(e))
    torch.save(target_returns, replay_folder + "/returns_" + str(e))

# torch.save(model, "model.pt")
