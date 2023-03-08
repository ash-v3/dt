import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
from datetime import datetime
from dt import DecisionTransformer
from replay_buffer import ReplayBuffer

TARGET_RETURN = 10000
EPOCHS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("CarRacing-v2")
state_dim = env.observation_space.shape
image_dim = [state_dim[2], state_dim[0], state_dim[1]]
act_dim = env.action_space.shape[0]
encoding_dim = 768

model = DecisionTransformer(state_dim=encoding_dim, act_dim=act_dim)
replay_buffer = ReplayBuffer()

observation, info = env.reset()

for e in range(EPOCHS):
    state, _ = env.reset()
    actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
    encoding = model.proc_state(observation)
    states = encoding.reshape(1, 1, encoding_dim).to(device=device, dtype=torch.float32)
    target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)
    with torch.inference_mode():
        state_preds, action_preds, return_preds = model(
            states=states,
            actions=actions,
            rewards=rewards,
            returns_to_go=target_return,
            timesteps=timesteps,
            attention_mask=attention_mask,
            return_dict=False,
        )

    terminated = truncated = False
    while not (terminated or truncated):
        with torch.inference_mode():
            state_preds, action_preds, return_preds = model(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=target_return,
                timesteps=timesteps,
                attention_mask=attention_mask,
                return_dict=False,
            )

            action = action_preds.squeeze().numpy()

        observation, reward, terminated, truncated, info = env.step(action)

        actions = torch.cat([actions, torch.from_numpy(action).reshape([1, 1, act_dim])], dim=1)
        rewards = torch.cat([rewards, torch.tensor([reward]).unsqueeze(0)], dim=0)
        encoding = model.proc_state(observation)
        print(target_return.shape, return_preds.shape)
        target_return = torch.cat([target_return, return_preds], dim=1)
        states = torch.cat([states, encoding.reshape([1, 1, encoding_dim])], dim=1)

        if random.randint(1, 20) > 19:
            terminated = True

    replay_buffer.cat({"states": states, "actions": actions, "rewards": rewards})
    print(replay_buffer.buffer[0])

now = datetime.now()
torch.save(replay_buffer, "saved_buffer" + now.strftime("-%H-%M-%S") + ".pt")
torch.save(model, "model.pt")
