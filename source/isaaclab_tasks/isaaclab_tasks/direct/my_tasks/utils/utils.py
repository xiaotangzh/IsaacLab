import sys
sys.path.append("D:/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/my_tasks")
import torch
from agents.base_agent import BaseAgent
from skrl.models.torch import Model
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LinearSegmentedColormap
from isaaclab.utils.math import quat_rotate

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

            # record the environments' transitions
            # agent.record_transition(
            #     states=states,
            #     actions=actions,
            #     rewards=rewards,
            #     next_states=next_states,
            #     terminated=terminated,
            #     truncated=truncated,
            #     infos=infos,
            #     timestep=timestep,
            #     timesteps=timesteps,
            # )

        # post-interaction (update is here)
        # agent.post_interaction(timestep=timestep, timesteps=timesteps)

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

def pairwise_joint_distance_weight(x, sqrt=True, upper_bound=1.5):
    assert len(x.shape) <= 2, "x must be 1D or 2D tensor"

    value = 1 - x / upper_bound
    value = torch.clamp(value, 0.0, 1.0)
    if sqrt: value = torch.sqrt(value)

    return torch.mean(value, dim=1) if len(x.shape) == 2 else value

def plot_pairwise_joint_distance_heatmap(x, key_names=None, upper_bound=1.5): # x: [keys * keys]
    assert len(x.shape) <= 1, "x must be 1D tensor"
    keys = int(math.sqrt(x.shape[0]))
    value = pairwise_joint_distance_weight(x, sqrt=False).detach().cpu().numpy()
    heatmap = value.reshape(keys, keys)
    
    white_red_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
    plt.clf() 
    plt.imshow(heatmap, cmap=white_red_cmap, interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Weight')

    if key_names is not None:
        assert len(key_names) == keys, "Length of key_names must match keys"
        plt.xticks(ticks=np.arange(keys), labels=key_names, rotation=0)
        plt.yticks(ticks=np.arange(keys), labels=key_names)

    plt.title(f"Pairwise Joint Distance Heatmap: $x \\to 1 - x / {upper_bound}$")
    plt.xlabel("Key Index")
    plt.ylabel("Key Index")
    plt.pause(0.001) 

def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_rotate(q, ref_tangent)
    normal = quat_rotate(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

def check_nan(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1: return torch.isnan(tensor)
    return torch.isnan(tensor).any(dim=-1)

def check_any_nan(tensor: torch.Tensor) -> torch.Tensor:
    return torch.any(check_nan(tensor))

def compute_action_clip(clip: list, current_steps: int):
    clip_min, clip_max, annealing_steps = clip[0], clip[1], clip[2]
    if current_steps >= annealing_steps: return clip_max
    return clip_min + (clip_max - clip_min) * (current_steps / annealing_steps)

# discarded
def plot_pairwise_joint_distance_weight():
    x, y, x_max, y_max, bias, m, n = pairwise_joint_distance_weight()

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title(f"Plot of y = {m} / ({n}x+1)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(1, 2.1)
    plt.grid(True)
    plt.axvline(x=x_max, color='red', linestyle='--', label=f'x = {x_max}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_pairwise_joint_distance_weight()