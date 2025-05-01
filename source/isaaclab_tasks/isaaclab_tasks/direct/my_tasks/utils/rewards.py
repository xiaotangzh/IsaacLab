import torch
from isaaclab_tasks.direct.my_tasks.utils.reward_utils import *
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_tasks.direct.my_tasks.env import Env  # avoid circular import error

# body positions imitation
def reward_imitation(env: "Env", loss_function: str = "MSE") -> torch.Tensor:
    obs_pos, obs_rot = env.robot1.data.body_pos_w, env.robot1.data.body_quat_w # (num_envs, bodies, 3/4)
    ref_pos, ref_rot = env.ref_state_buffer_1["body_pos"][env.episode_length_buf] + env.scene.env_origins.unsqueeze(1), env.ref_state_buffer_1["body_rot"][env.episode_length_buf]

    if env.robot2:
        obs2_pos, obs2_rot = env.robot2.data.body_pos_w, env.robot2.data.body_quat_w
        ref2_pos, ref2_rot = env.ref_state_buffer_2["body_pos"][env.episode_length_buf] + env.scene.env_origins.unsqueeze(1), env.ref_state_buffer_2["body_rot"][env.episode_length_buf]
        obs_pos, obs_rot = torch.cat([obs_pos, obs2_pos], dim=1), torch.cat([obs_rot, obs2_rot], dim=1) # (num_envs, 2 * bodies, 3/4)
        ref_pos, ref_rot = torch.cat([ref_pos, ref2_pos], dim=1), torch.cat([ref_rot, ref2_rot], dim=1) # (num_envs, 2 * bodies, 3/4)

    env.green_markers_small.visualize(translations=ref_pos.reshape(-1, 3))
    env.red_markers_small.visualize(translations=obs_pos.reshape(-1, 3))
    
    match loss_function:
        case "MSE":
            loss = torch.mean(obs_pos.reshape([env.num_envs, -1]) - 
                                ref_pos.reshape([env.num_envs, -1]), dim=1)
            loss = torch.abs(loss)
            reward = torch.clamp(2 * (0.5 - loss), min=0.0, max=1.0)
        case "L2":
            loss = torch.mean(torch.norm(obs_pos - ref_pos, dim=-1), dim=1)
            reward = 2 * torch.clamp(0.5 - loss, min=0.0, max=1.0)
    # print(f"Imitation reward: {torch.mean(reward)}")
    return reward

def reward_zero(env: "Env") -> torch.Tensor:
    return torch.zeros((env.num_envs,), dtype=torch.float32, device=env.sim.device)

def reward_energy_penalty(env: "Env", weight: float=1.0) -> torch.Tensor:
    dof_forces = env.robot1.data.applied_torque
    dof_vel = env.robot1.data.joint_vel

    power = torch.abs(torch.multiply(dof_forces, dof_vel)).sum(dim=-1)

    # normalize
    norm = 10000
    power = torch.clip(power / norm, min=0.0, max=1.0) * weight
    pow_rew = -power

    # print(f"Energy penalty: {torch.mean(pow_rew)}")
    return pow_rew


def reward_stand_forward(env: "Env") -> torch.Tensor:
    angle_offset = compute_angle_offset(env, "forward", env.robot1) 
    reward = torch.abs(angle_offset)
    if env.robot2: 
        angle_offset2 = compute_angle_offset(env, "forward", env.robot2)
        reward = (reward + torch.abs(angle_offset2)) / 2
    # print(f"stand forward reward: {torch.mean(reward)}")
    return reward

def reward_min_vel(env: "Env") -> torch.Tensor: # [0, 3]
    reward = torch.mean(env.robot1.data.joint_vel, dim=-1)
    if env.robot2: 
        reward = (reward + torch.mean(env.robot2.data.joint_vel, dim=-1)) / 2
    reward = torch.clip((reward * (-1) + 3) / 3, min=0, max=1)
    # print(f"min vel reward: {torch.mean(reward)}")
    return reward

def reward_stand(env: "Env") -> torch.Tensor:
    reward = torch.clip(env.robot1.data.body_pos_w[:, env.ref_body_index, 2], min=0, max=env.robot1.data.default_root_state[0, 2].item())
    if env.robot2:
        reward = (reward + torch.clip(env.robot2.data.body_pos_w[:, env.ref_body_index, 2], min=0, max=env.robot2.data.default_root_state[0, 2].item())) / 2
    # print(f"stand reward: {torch.mean(reward)}")
    return reward

def reward_com(env: "Env") -> torch.Tensor:
    env.com_robot1 = compute_whole_body_com(env, env.robot1)
    reward = 1 - torch.mean(torch.abs(env.com_robot1 - env.default_com))
    if env.robot2:
        env.com_robot2 = compute_whole_body_com(env, env.robot2)
        reward2 = 1 - torch.mean(torch.abs(env.com_robot2 - env.default_com))
        reward = (reward + reward2) / 2
    # print(f"center of mass reward: {torch.mean(reward)}")
    return reward

def reward_com_acc(env: "Env", decay: float=0.01) -> torch.Tensor:
    reward = torch.clip(torch.exp(-decay * torch.mean(torch.abs(env.com_acc_robot1))), min=0.0, max=1.0)
    if env.robot2:
        reward2 = torch.clip(torch.exp(-decay * torch.mean(torch.abs(env.com_acc_robot2))), min=0.0, max=1.0)
        reward = (reward + reward2)
    # print(f"center of mass acc reward: {reward}")
    return reward

