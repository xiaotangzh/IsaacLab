import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_rotate
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

def compute_angle_offset(env: "Env", target_direction, robot: "Articulation") -> torch.Tensor:
    if target_direction == "forward":
        target_direction = torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
        idx = 0
    elif target_direction == "leftward":
        target_direction = torch.tensor([0.0, 1.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
        idx = 1
    elif target_direction == "upward":
        target_direction = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
        idx = 2

    current_quat = robot.data.body_quat_w[:, env.ref_body_index]
    current_quat = current_quat / torch.norm(current_quat, dim=-1, keepdim=True)
    current_direction = quat_rotate(current_quat, target_direction)
    angle_offset = current_direction[:, idx]  # [-1, 1] from opposite to same direction as target
    return angle_offset