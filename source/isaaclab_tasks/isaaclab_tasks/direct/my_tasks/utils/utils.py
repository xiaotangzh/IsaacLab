import sys
import torch
from agents.base_agent import BaseAgent
from skrl.models.torch import Model
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LinearSegmentedColormap


def evaluate(agent: BaseAgent, env, args):
    agent.set_running_mode("eval")
    disable_grads(agent)
    timestep, timesteps = 0, 100000
    # reset env
    states, infos = env.reset()
    while(True):
        # pre-interaction
        agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with torch.no_grad():
            # compute actions
            actions = agent.act(states, timestep=timestep, timesteps=100000)[0]

            # step the environments
            next_states, rewards, terminated, truncated, infos = env.step(actions)

            # render scene
            if not args.headless:
                env.render()

        # reset environments
        states = next_states

def disable_grads(agent: BaseAgent):
    for k, v in vars(agent).items():
        if isinstance(v, Model):
            for p in v.parameters():
                p.requires_grad = False

def get_algorithm(task: str) -> str:
    if "AMP" in task:
        return "AMP"
    elif "PPO" in task:
        return "PPO"
    elif "MOE" in task:
        return "MOE"
    else:
        return ""


def check_nan(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1: return torch.isnan(tensor)
    return torch.isnan(tensor).any(dim=-1)

def check_any_nan(tensor: torch.Tensor) -> torch.Tensor:
    return torch.any(check_nan(tensor))
