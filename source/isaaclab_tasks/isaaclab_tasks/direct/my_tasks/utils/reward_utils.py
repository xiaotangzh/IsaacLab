import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_rotate
import torch.nn.functional as F
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_tasks.direct.my_tasks.env import Env  # avoid circular import error
    from isaaclab.assets import Articulation

def compute_relative_body_positions(env: "Env", source: "Articulation", target: "Articulation") -> torch.Tensor:
    source_body_positions = source.data.body_pos_w # (envs, body num, 3)
    target_root_positions = target.data.body_pos_w[:,env.ref_body_index] # (envs, 3)
    target_root_rotations = target.data.body_quat_w[:,env.ref_body_index] # (envs, 4)
    return math_utils.transform_points(points=source_body_positions, pos=target_root_positions, quat=target_root_rotations)

def compute_coms(env: "Env"):
    current_com_robot1 = compute_whole_body_com(env, env.robot1)
    current_com_vel_robot1 = (current_com_robot1 - env.com_robot1) / env.cfg.dt
    current_com_acc_robot1 = (current_com_vel_robot1 - env.com_vel_robot1) / env.cfg.dt
    # update coms    
    env.com_robot1, env.com_vel_robot1, env.com_acc_robot1 = current_com_robot1, current_com_vel_robot1, current_com_acc_robot1

    if env.robot2:
        current_com_robot2 = compute_whole_body_com(env, env.robot2)
        current_com_vel_robot2 = (current_com_robot2 - env.com_robot2) / env.cfg.dt
        current_com_acc_robot2 = (current_com_vel_robot2 - env.com_vel_robot2) / env.cfg.dt
        # update coms    
        env.com_robot2, env.com_vel_robot2, env.com_acc_robot2 = current_com_robot2, current_com_vel_robot2, current_com_acc_robot2

    # visualize markers
    translations = torch.cat([env.com_robot1, env.com_robot2], dim=0) if env.robot2 else env.com_robot1
    scales = 1 + torch.sigmoid(torch.norm(translations, dim=1, keepdim=True)).repeat(1, 3)
    env.red_markers.visualize(translations=env.com_robot1, scales=scales)

def compute_whole_body_com(env: "Env", robot: "Articulation") -> torch.Tensor:
    # whole_body_com = Σ(bone_mass × bone_com) / total_mass
    body_masses = robot.data.default_mass.to(env.device) # [num_envs, num_bodies]
    total_mass = body_masses.sum(dim=1, keepdim=True)  # [num_envs, 1]
    body_com_positions = robot.data.body_com_pos_w # [num_envs, num_bodies, 3]
    weighted_positions = body_com_positions * body_masses.unsqueeze(-1)  # [num_envs, num_bodies, 3]
    whole_body_com = weighted_positions.sum(dim=1) / total_mass  # [num_envs, 3]
    return whole_body_com # [num_envs, 3]

def compute_zmp(env: "Env", robot: "Articulation"):
    # Zero Moment Point

    # 获取各连杆参数
    masses = robot.data.default_mass.to(env.device)  # [num_instances, num_bodies]
    poses = robot.data.body_com_state_w[..., :3] # [..., num_bodies, 3]
    accs = robot.data.body_acc_w  # [..., num_bodies, 6]
    
    # 提取线性加速度
    lin_acc = accs[..., 0:3]  # [ẍ, ÿ, z̈]
    
    # 计算分子分母
    numerator_x = torch.sum(masses * (9.8 + lin_acc[..., 2]) * poses[..., 0] - 
                        masses * lin_acc[..., 0] * poses[..., 2], dim=-1)
    numerator_y = torch.sum(masses * (9.8 + lin_acc[..., 2]) * poses[..., 1] - 
                        masses * lin_acc[..., 1] * poses[..., 2], dim=-1)
    denominator = torch.sum(masses * (9.8 + lin_acc[..., 2]), dim=-1)
    
    # 计算ZMP坐标
    zmp_x = numerator_x / denominator
    zmp_y = numerator_y / denominator
    
    return torch.stack([zmp_x, zmp_y], dim=-1)  # [num_instances, 2]

def get_unit_vector(env: "Env", target_direction):
    if target_direction == "forward":
        target_direction = torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0)
        idx = 0
    elif target_direction == "leftward":
        target_direction = torch.tensor([0.0, 1.0, 0.0], device=env.device).unsqueeze(0)
        idx = 1
    elif target_direction == "upward":
        target_direction = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0)
        idx = 2
    return target_direction, idx

def transform_quat_to_target_direction(current_quat, target_direction) -> torch.Tensor:
    current_quat = current_quat / torch.norm(current_quat, dim=-1, keepdim=True)
    current_direction = quat_rotate(current_quat, target_direction)
    return current_direction

def compute_angle_offset(env: "Env", target_direction, robot: "Articulation", angle) -> torch.Tensor:
    target_direction, idx = get_unit_vector(env, target_direction)
    target_direction = target_direction.expand(env.num_envs, -1)
    current_direction = transform_quat_to_target_direction(robot.data.body_quat_w[:, env.ref_body_index], target_direction)
    angle_offset = current_direction[:, idx]  # [-1, 1] from opposite to same direction as target
    return (angle_offset < angle)

def compute_stuck(env: "Env", robot1: "Articulation", robot2: "Articulation"):
    avg_body_pos = torch.mean(torch.norm(robot1.data.body_pos_w - robot2.data.body_pos_w, dim=-1), dim=-1)
    avg_body_vel = torch.mean(torch.norm(torch.cat([robot1.data.body_vel_w, robot2.data.body_vel_w], dim=-1), dim=-1), dim=-1)

    up, _ = get_unit_vector(env, "upward")
    up = up.expand(env.num_envs, -1)
    up1 = transform_quat_to_target_direction(robot1.data.body_quat_w[:, env.ref_body_index], up)
    up2 = transform_quat_to_target_direction(robot2.data.body_quat_w[:, env.ref_body_index], up)
    angle = F.cosine_similarity(up1, up2, dim=-1) # [-1, 1] from large to small angle
    angle_rad = torch.acos(torch.clamp(angle, -1.0, 1.0))  # [0, π]
    angle_deg = angle_rad * (180.0 / math.pi)  # [0, 180]

    # print(avg_pos_too_close, avg_vel_too_small)
    stucked = (avg_body_pos < 1.4) & (avg_body_vel < 8) & (angle_deg > 25)
    # print(stucked.shape)

    # stucked.nonzero(as_tuple=False).squeeze(-1)
    # print(stucked.shape)
    return stucked

def compute_died_height(env: "Env", robot: "Articulation", termination_heights):
    died_height = robot.data.body_pos_w[:, env.early_termination_body_indexes, 2] < termination_heights
    died_height = torch.max(died_height, dim=1).values
    return died_height