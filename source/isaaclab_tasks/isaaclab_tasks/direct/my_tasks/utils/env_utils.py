from isaaclab.utils.math import quat_rotate
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
from matplotlib.colors import LinearSegmentedColormap
import isaaclab.utils.math as math_utils
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_tasks.direct.my_tasks.env import Env  # avoid circular import error
    from isaaclab.assets import Articulation
    from isaaclab_tasks.direct.my_tasks.motions.motion_loader import MotionLoader, MotionLoaderHumanoid28, MotionLoaderSMPL

def compute_obs(env: "Env",
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_position: torch.Tensor,
    root_rotation: torch.Tensor,
    root_linear_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    relative_pose: torch.Tensor | None=None,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_position[:, 2:3], # rigid body values are in "world" space !
            quaternion_to_tangent_and_normal(root_rotation),
            root_linear_velocity,
            root_angular_velocity,
        ),
        dim=-1,
    )
    
    if relative_pose is not None: 
        relative_pose = relative_pose.reshape(obs.shape[0], -1)
        obs = torch.cat((obs, relative_pose), dim=-1)

    return obs

def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_rotate(q, ref_tangent)
    normal = quat_rotate(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

def compute_action_clip(clip: list | float, current_steps: int):
    if isinstance(clip, float): return clip
    clip_min, clip_max, annealing_steps = clip[0], clip[1], clip[2]
    if current_steps >= annealing_steps: return clip_max
    return clip_min + (clip_max - clip_min) * (current_steps / annealing_steps)

def reset_reference_buffer(env: "Env", motion_loader, ref_state_buffer: dict, env_ids: torch.Tensor | None=None):
    env_ids = env.robot1._ALL_INDICES if env_ids is None else env_ids
    num_samples = 1 #env_ids.shape[0]
    
    # sample reference actions
    (
        ref_dof_positions,
        ref_dof_velocities,
        ref_body_positions,
        ref_body_rotations,
        ref_body_linear_velocities,
        ref_body_angular_velocities,
    ) = motion_loader.get_all_references(num_samples)
    
    # ref_root_state = env.robot1.data.default_root_state[env_ids].unsqueeze(1).expand(-1, ref_dof_positions.shape[1], -1).clone()
    ref_root_state = torch.zeros([num_samples, ref_dof_positions.shape[1], 13], device=env.device)
    ref_root_state[:, :, 0:3] = ref_body_positions[:, :, env.motion_ref_body_index] #+ env.scene.env_origins[env_ids].unsqueeze(1)
    ref_root_state[:, :, 3:7] = ref_body_rotations[:, :, env.motion_ref_body_index]
    ref_root_state[:, :, 7:10] = ref_body_linear_velocities[:, :, env.motion_ref_body_index]
    ref_root_state[:, :, 10:13] = ref_body_angular_velocities[:, :, env.motion_ref_body_index]
    
    # set reference buffer
    _ = motion_loader.get_all_references(num_samples)
    ref_state_buffer.update({
        "root_state": ref_root_state.squeeze(0),
        "dof_pos": ref_dof_positions[:, :, env.motion_dof_indexes].squeeze(0),
        "dof_vel": ref_dof_velocities[:, :, env.motion_dof_indexes].squeeze(0),
        "body_pos": ref_body_positions[:, :, env.motion_body_indexes].squeeze(0),
        "body_rot": ref_body_rotations[:, :, env.motion_body_indexes].squeeze(0),
    })
    return


def update_amp_buffer(
    env: "Env",
    amp_obs,
    amp_observation_buffer,
    num_amp_observations,
):
    # amp_buffer: [num_envs, 1 or 2 robot, num_amp_observations, amp_observation_space]

    if env.robot2 is None:
        assert amp_observation_buffer.shape[1] == 1

        # update AMP observation history (pop out)
        for i in reversed(range(num_amp_observations - 1)):
            amp_observation_buffer[:, :, i + 1] = amp_observation_buffer[:, :, i]

        # update AMP observation history (push in)
        amp_observation_buffer[:, 0, 0] = amp_obs.view(env.num_envs, -1) 

        return amp_observation_buffer
    else:
        assert amp_observation_buffer.shape[1] == 2
        amp_obs_1, amp_obs_2 = torch.chunk(amp_obs, 2, dim=0)

        # update AMP observation history (pop out)
        for i in reversed(range(num_amp_observations - 1)):
            amp_observation_buffer[:, :, i + 1] = amp_observation_buffer[:, :, i]

        # update AMP observation history (push in)
        amp_observation_buffer[:, 0, 0] = amp_obs_1.view(env.num_envs, -1)
        amp_observation_buffer[:, 1, 0] = amp_obs_2.view(env.num_envs, -1)

        return amp_observation_buffer
