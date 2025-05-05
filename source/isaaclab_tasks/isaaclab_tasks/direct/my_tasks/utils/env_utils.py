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

def precompute_relative_body_positions(env: "Env", source: "MotionLoader", target: "MotionLoader") -> torch.Tensor:
    source_body_positions = source.get_all_references()[2][0] # (frames, body num, 3)
    target_root_positions = target.get_all_references()[2][0][:,env.motion_ref_body_index] # (frames, 3)
    target_root_rotations = target.get_all_references()[3][0][:,env.motion_ref_body_index] # (frames, 4)
    return math_utils.transform_points(points=source_body_positions, pos=target_root_positions, quat=target_root_rotations)

def compute_interaction_env_weight(env: "Env", ego: "MotionLoader", target: "MotionLoader") -> torch.Tensor:
    body_positions_1 = ego.get_all_references()[2][0, :, env.motion_body_indexes] # [frames, body num, 3]
    body_positions_2 = target.get_all_references()[2][0, :, env.motion_body_indexes] # [frames, body num, 3]
    instances = body_positions_1.shape[0]

    # key bodies
    body_positions_1 = body_positions_1[:, env.key_body_indexes] # [frames or envs, key body num, 3]
    body_positions_2 = body_positions_2[:, env.key_body_indexes] # [frames or envs, key body num, 3]

    # calculate pairwise distance
    body_positions_1_expand = body_positions_1.unsqueeze(2)  # [frames or envs, body_num, 1, 3]
    body_positions_2_expand = body_positions_2.unsqueeze(1)  # [frames or envs, 1, body_num, 3]
    diff = body_positions_1_expand - body_positions_2_expand  # [frames or envs, body_num, body_num, 3]
    pairwise_joint_distance = torch.norm(diff, dim=-1).view(instances, -1) # [frames or envs, body_num * body_num]

    # compute weight for each env
    interaction_reward_weights = compute_pairwise_joint_distance_weight(pairwise_joint_distance, sqrt=env.pjd_cfg["sqrt"], upper_bound=env.pjd_cfg["upper_bound"]).view(instances, -1)
    match env.pjd_cfg["weight_method"]:
        case "mean": interaction_reward_weights = torch.mean(interaction_reward_weights, dim=1)
        case "max" : interaction_reward_weights = torch.max(interaction_reward_weights, dim=1)[0]
    return interaction_reward_weights

# test: pjd1 pairwise joint distance
def compute_pairwise_joint_distance(env: "Env", ego: Union["MotionLoader", "Articulation"], target: Union["MotionLoader", "Articulation"]):
    cls1 = type(ego).__name__
    cls2 = type(target).__name__

    if "MotionLoader" in cls1 and "MotionLoader" in cls2:
        body_positions_1 = ego.get_all_references()[2][0, :, env.motion_body_indexes] # [frames, body num, 3]
        body_positions_2 = target.get_all_references()[2][0, :, env.motion_body_indexes] # [frames, body num, 3]
    elif "Articulation" in cls1 and "Articulation" in cls2:
        body_positions_1 = ego.data.body_pos_w # [envs, body num, 3]
        body_positions_2 = target.data.body_pos_w # [envs, body num, 3]
    instances = body_positions_1.shape[0]

    # key bodies
    body_positions_1 = body_positions_1[:, env.key_body_indexes] # [frames or envs, key body num, 3]
    body_positions_2 = body_positions_2[:, env.key_body_indexes] # [frames or envs, key body num, 3]

    # calculate pairwise distance
    body_positions_1_expand = body_positions_1.unsqueeze(2)  # [frames or envs, body_num, 1, 3]
    body_positions_2_expand = body_positions_2.unsqueeze(1)  # [frames or envs, 1, body_num, 3]
    diff = body_positions_1_expand - body_positions_2_expand  # [frames or envs, body_num, body_num, 3]
    pairwise_joint_distance = torch.norm(diff, dim=-1).view(instances, -1) # [frames or envs, body_num * body_num]
    
    if env.pjd_cfg["weighted"]: 
        pairwise_joint_distance = compute_pairwise_joint_distance_weight(pairwise_joint_distance, sqrt=env.pjd_cfg["sqrt"], upper_bound=env.pjd_cfg["upper_bound"]) # [frames or envs, body_num * body_num]

    return pairwise_joint_distance


# test: pjd2  relative body positions + dof velocities
# def compute_pairwise_joint_distance(env: "Env", ego: Union["MotionLoader", "Articulation"], target: Union["MotionLoader", "Articulation"], compute_weight: bool=False) -> torch.Tensor:
#     '''
#     All DoF velocities of the other robot + Pairwise joint relative position (keys * keys * 3)
#     '''
#     cls1 = type(ego).__name__
#     cls2 = type(target).__name__

#     if "MotionLoader" in cls1 and "MotionLoader" in cls2:
#         body_positions_1 = ego.get_all_references()[2][0, :, env.motion_body_indexes] # [frames, body num, 3]
#         body_positions_2 = target.get_all_references()[2][0, :, env.motion_body_indexes] # [frames, body num, 3]
#         dof_velocities_2 = target.get_all_references()[1][0, :, env.motion_dof_indexes] # [frames, dof num]
#     elif "Articulation" in cls1 and "Articulation" in cls2:
#         body_positions_1 = ego.data.body_pos_w # [envs, body num, 3]
#         body_positions_2 = target.data.body_pos_w # [envs, body num, 3]
#         dof_velocities_2 = target.data.joint_vel # [envs, dof num]
#     instances, keys = body_positions_1.shape[0], len(env.key_body_indexes)

#     # key bodies
#     body_positions_1 = body_positions_1[:, env.key_body_indexes] # [frames or envs, key body num, 3]
#     body_positions_2 = body_positions_2[:, env.key_body_indexes] # [frames or envs, key body num, 3]

#     # calculate pairwise distance
#     body_positions_1_expand = body_positions_1.unsqueeze(2)  # [frames or envs, body_num, 1, 3]
#     body_positions_2_expand = body_positions_2.unsqueeze(1)  # [frames or envs, 1, body_num, 3]
#     diff = body_positions_1_expand - body_positions_2_expand  # [frames or envs, body_num, body_num, 3]
#     relative_positions = diff.view(instances, -1)  # [frames or envs, body_num * body_num * 3]
#     pairwise_joint_distance = torch.norm(relative_positions.reshape(instances, -1, 3), dim=-1)

#     # concatenate with dof velocities
#     interaction = torch.cat([dof_velocities_2, pairwise_joint_distance], dim=-1)  # [frames or envs, interaction_space]

#     return interaction

# test: pjd3  relative body positions + relative body velocities
# def compute_pairwise_joint_distance(env: "Env", ego: Union["MotionLoader", "Articulation"], target: Union["MotionLoader", "Articulation"], compute_weight: bool=False) -> torch.Tensor:
#     cls1 = type(ego).__name__
#     cls2 = type(target).__name__

#     if "MotionLoader" in cls1 and "MotionLoader" in cls2:
#         body_positions_1 = ego.get_all_references()[2][0, :, env.motion_body_indexes] # [frames, body num, 3]
#         body_positions_2 = target.get_all_references()[2][0, :, env.motion_body_indexes] # [frames, body num, 3]
#         body_velocities_1 = ego.get_all_references()[4][0, :, env.motion_body_indexes] # [frames, body num, 3]
#         body_velocities_2 = target.get_all_references()[4][0, :, env.motion_body_indexes] # [frames, body num, 3]
#     elif "Articulation" in cls1 and "Articulation" in cls2:
#         body_positions_1 = ego.data.body_pos_w # [envs, body num, 3]
#         body_positions_2 = target.data.body_pos_w # [envs, body num, 3]
#         body_velocities_1 = ego.data.body_lin_vel_w # [envs, dof num]
#         body_velocities_2 = target.data.body_lin_vel_w # [envs, dof num]
#     instances = body_positions_1.shape[0]

#     # key bodies
#     body_positions_1 = body_positions_1[:, env.key_body_indexes] # [frames or envs, key body num, 3]
#     body_positions_2 = body_positions_2[:, env.key_body_indexes] # [frames or envs, key body num, 3]
#     body_velocities_1 = body_velocities_1[:, env.key_body_indexes] # [frames or envs, key body num, 3]
#     body_velocities_2 = body_velocities_2[:, env.key_body_indexes] # [frames or envs, key body num, 3]

#     # calculate body position
#     body_positions_1_expand = body_positions_1.unsqueeze(2)  # [frames or envs, body_num, 1, 3]
#     body_positions_2_expand = body_positions_2.unsqueeze(1)  # [frames or envs, 1, body_num, 3]
#     relative_positions = body_positions_1_expand - body_positions_2_expand  # [frames or envs, body_num, body_num, 3]
#     pairwise_joint_distance = torch.norm(relative_positions, dim=-1)  # [frames or envs, body_num, body_num]
#     pairwise_joint_distance = pairwise_joint_distance.view(instances, -1)  # [frames or envs, body_num * body_num]

#     # calculate relative body velocities
#     body_velocities_1_expand = body_velocities_1.unsqueeze(2)  # [frames or envs, body_num, 1, 3]
#     body_velocities_2_expand = body_velocities_2.unsqueeze(1)  # [frames or envs, 1, body_num, 3]
#     relative_velocities = body_velocities_1_expand - body_velocities_2_expand  # [frames or envs, body_num, body_num, 3]

#     # combine
#     interaction = torch.cat([relative_positions.view(instances, -1), relative_velocities.view(instances, -1)], dim=-1)  # [frames or envs, interaction_space]

#     return interaction

# test: pjd4 relative body positions + pairwise joint distance
# def compute_pairwise_joint_distance(env: "Env", ego: Union["MotionLoader", "Articulation"], target: Union["MotionLoader", "Articulation"]) -> torch.Tensor:
#     cls1 = type(ego).__name__
#     cls2 = type(target).__name__

#     if "MotionLoader" in cls1 and "MotionLoader" in cls2:
#         body_positions_1 = ego.get_all_references()[2][0, :, env.motion_body_indexes] # [frames, body num, 3]
#         body_positions_2 = target.get_all_references()[2][0, :, env.motion_body_indexes] # [frames, body num, 3]
#     elif "Articulation" in cls1 and "Articulation" in cls2:
#         body_positions_1 = ego.data.body_pos_w # [envs, body num, 3]
#         body_positions_2 = target.data.body_pos_w # [envs, body num, 3]
#     instances = body_positions_1.shape[0]

#     # key bodies
#     ref_body_position_1 = body_positions_1[:, env.ref_body_index] # [frames or envs, 3]
#     body_positions_1 = body_positions_1[:, env.key_body_indexes] # [frames or envs, key body num, 3]
#     body_positions_2 = body_positions_2[:, env.key_body_indexes] # [frames or envs, key body num, 3]

#     # calculate relative body positions
#     body_positions_2_in_1 = body_positions_2 - ref_body_position_1.unsqueeze(1) # [frames or envs, key body num, 3]

#     # calculate pairwise joint distance
#     body_positions_1_expand = body_positions_1.unsqueeze(2)  # [frames or envs, body_num, 1, 3]
#     body_positions_2_expand = body_positions_2.unsqueeze(1)  # [frames or envs, 1, body_num, 3]
#     relative_positions = body_positions_1_expand - body_positions_2_expand  # [frames or envs, body_num, body_num, 3]
#     pairwise_joint_distance = torch.norm(relative_positions, dim=-1).view(instances, -1)  # [frames or envs, body_num * body_num]
#     if env.pjd_cfg["weighted"]: 
#         pairwise_joint_distance = compute_pairwise_joint_distance_weight(pairwise_joint_distance, sqrt=env.pjd_cfg["sqrt"], upper_bound=env.pjd_cfg["upper_bound"]) # [frames or envs, body_num * body_num]

#     # combine
#     interaction = torch.cat([body_positions_2_in_1.view(instances, -1), pairwise_joint_distance.view(instances, -1)], dim=-1)  # [frames or envs, interaction_space]

#     return interaction

def compute_pairwise_joint_distance_weight(x, sqrt=True, upper_bound=1.0):
    '''
    Compute weighted pairwise joint distance
    '''
    assert len(x.shape) <= 2, "x must be 1D or 2D tensor"

    value = 1 - x / upper_bound
    value = torch.clamp(value, 0.0, 1.0)
    if sqrt: value = value**2 #torch.sqrt(value)

    return value

def animate_pairwise_joint_distance_heatmap(x, key_names=None, sqrt=False, upper_bound=1.0): # x: [keys * keys]
    assert len(x.shape) <= 1, "x must be 1D tensor"

    keys = int(math.sqrt(x.shape[0]))
    value = compute_pairwise_joint_distance_weight(x, sqrt=sqrt).detach().cpu().numpy()
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

def update_amp_buffer(
    env: "Env",
    amp_obs,
    amp_observation_buffer,
    num_amp_observations,
):
    if env.robot2 is None:
        assert amp_observation_buffer.shape[1] == 1
        # update AMP observation history (pop out)
        for i in reversed(range(num_amp_observations - 1)):
            amp_observation_buffer[:, :, i + 1] = amp_observation_buffer[:, :, i]
        # update AMP observation history (push in)
        amp_observation_buffer[:, :, 0] = amp_obs # buffer: [num_envs, num_amp_observations, amp_observation_space]
        return amp_observation_buffer
    else:
        assert amp_observation_buffer.shape[1] == 2
        amp_obs_1, amp_obs_2 = torch.chunk(amp_obs, 2, dim=0)
        for i in reversed(range(num_amp_observations - 1)):
            amp_observation_buffer[:, :, i + 1] = amp_observation_buffer[:, :, i]
        amp_observation_buffer[:, 0, 0] = amp_obs_1 
        amp_observation_buffer[:, 1, 0] = amp_obs_2
        return amp_observation_buffer

def plot_pairwise_joint_distance_weight():
    x = np.linspace(0, 2.0, 100)
    upper_bound = 1.5
    y = compute_pairwise_joint_distance_weight(torch.from_numpy(x), sqrt=True, upper_bound=upper_bound).numpy()

    plt.figure(figsize=(4, 6))
    plt.plot(x, y)
    plt.title(f"Plot of y = 1 - x / {upper_bound}")
    plt.xlabel("Joint Distance")
    plt.ylabel("Weight")
    plt.ylim(0, 1.1)
    plt.grid(True)
    # plt.axvline(x=x_max, color='red', linestyle='--', label=f'x = {x_max}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_pairwise_joint_distance_weight()