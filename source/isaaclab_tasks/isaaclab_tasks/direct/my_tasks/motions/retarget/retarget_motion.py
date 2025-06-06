# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# from isaacgym.torch_utils import *
from hmac import new
from shlex import join
import sys
import torch
import json
import numpy as np

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from visualization import animate3D, plot_tpose

"""
This scripts shows how to retarget a motion clip from the source skeleton to a target skeleton.
Data required for retargeting are stored in a retarget config dictionary as a json file. This file contains:
  - source_motion: a SkeletonMotion npy format representation of a motion sequence. The motion clip should use the same skeleton as the source T-Pose skeleton.
  - target_motion_path: path to save the retargeted motion to
  - source_tpose: a SkeletonState npy format representation of the source skeleton in it's T-Pose state
  - target_tpose: a SkeletonState npy format representation of the target skeleton in it's T-Pose state (pose should match source T-Pose)
  - joint_mapping: mapping of joint names from source to target
  - rotation: root rotation offset from source to target skeleton (for transforming across different orientation axes), represented as a quaternion in XYZW order.
  - scale: scale offset from source to target skeleton
"""

VISUALIZE = True

SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]
SMPL_MUJOCO_NAMES = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
# SMPL_MUJOCO_NAMES = ['Pelvis', 'Torso', 'Head', 'R_Shoulder', 'R_Elbow', 'R_Hand', 'L_Shoulder', 'L_Elbow', 'L_Hand', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle']
body_names_humanoid28 = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'left_upper_arm', 'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']
# body_names_humanoid28 = ['pelvis', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot',  'right_upper_arm', 'right_lower_arm', 'right_hand', 'left_upper_arm', 'left_lower_arm', 'left_hand', 'torso', 'head',]

def project_joints(motion):
    right_upper_arm_id = motion.skeleton_tree._node_indices["right_upper_arm"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_lower_arm"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_upper_arm"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_lower_arm"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_thigh"]
    right_shin_id = motion.skeleton_tree._node_indices["right_shin"]
    right_foot_id = motion.skeleton_tree._node_indices["right_foot"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_thigh"]
    left_shin_id = motion.skeleton_tree._node_indices["left_shin"]
    left_foot_id = motion.skeleton_tree._node_indices["left_foot"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)
    
    # right leg
    right_thigh_pos = motion.global_translation[..., right_thigh_id, :]
    right_shin_pos = motion.global_translation[..., right_shin_id, :]
    right_foot_pos = motion.global_translation[..., right_foot_id, :]
    right_hip_rot = motion.local_rotation[..., right_thigh_id, :]
    right_knee_rot = motion.local_rotation[..., right_shin_id, :]
    
    right_leg_delta0 = right_thigh_pos - right_shin_pos
    right_leg_delta1 = right_foot_pos - right_shin_pos
    right_leg_delta0 = right_leg_delta0 / torch.norm(right_leg_delta0, dim=-1, keepdim=True)
    right_leg_delta1 = right_leg_delta1 / torch.norm(right_leg_delta1, dim=-1, keepdim=True)
    right_knee_dot = torch.sum(-right_leg_delta0 * right_leg_delta1, dim=-1)
    right_knee_dot = torch.clamp(right_knee_dot, -1.0, 1.0)
    right_knee_theta = torch.acos(right_knee_dot)
    right_knee_q = quat_from_angle_axis(torch.abs(right_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    right_knee_local_dir = motion.skeleton_tree.local_translation[right_foot_id]
    right_knee_local_dir = right_knee_local_dir / torch.norm(right_knee_local_dir)
    right_knee_local_dir_tile = torch.tile(right_knee_local_dir.unsqueeze(0), [right_knee_rot.shape[0], 1])
    right_knee_local_dir0 = quat_rotate(right_knee_rot, right_knee_local_dir_tile)
    right_knee_local_dir1 = quat_rotate(right_knee_q, right_knee_local_dir_tile)
    right_leg_dot = torch.sum(right_knee_local_dir0 * right_knee_local_dir1, dim=-1)
    right_leg_dot = torch.clamp(right_leg_dot, -1.0, 1.0)
    right_leg_theta = torch.acos(right_leg_dot)
    right_leg_theta = torch.where(right_knee_local_dir0[..., 1] >= 0, right_leg_theta, -right_leg_theta)
    right_leg_q = quat_from_angle_axis(right_leg_theta, right_knee_local_dir.unsqueeze(0))
    right_hip_rot = quat_mul(right_hip_rot, right_leg_q)
    
    # left leg
    left_thigh_pos = motion.global_translation[..., left_thigh_id, :]
    left_shin_pos = motion.global_translation[..., left_shin_id, :]
    left_foot_pos = motion.global_translation[..., left_foot_id, :]
    left_hip_rot = motion.local_rotation[..., left_thigh_id, :]
    left_knee_rot = motion.local_rotation[..., left_shin_id, :]
    
    left_leg_delta0 = left_thigh_pos - left_shin_pos
    left_leg_delta1 = left_foot_pos - left_shin_pos
    left_leg_delta0 = left_leg_delta0 / torch.norm(left_leg_delta0, dim=-1, keepdim=True)
    left_leg_delta1 = left_leg_delta1 / torch.norm(left_leg_delta1, dim=-1, keepdim=True)
    left_knee_dot = torch.sum(-left_leg_delta0 * left_leg_delta1, dim=-1)
    left_knee_dot = torch.clamp(left_knee_dot, -1.0, 1.0)
    left_knee_theta = torch.acos(left_knee_dot)
    left_knee_q = quat_from_angle_axis(torch.abs(left_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    left_knee_local_dir = motion.skeleton_tree.local_translation[left_foot_id]
    left_knee_local_dir = left_knee_local_dir / torch.norm(left_knee_local_dir)
    left_knee_local_dir_tile = torch.tile(left_knee_local_dir.unsqueeze(0), [left_knee_rot.shape[0], 1])
    left_knee_local_dir0 = quat_rotate(left_knee_rot, left_knee_local_dir_tile)
    left_knee_local_dir1 = quat_rotate(left_knee_q, left_knee_local_dir_tile)
    left_leg_dot = torch.sum(left_knee_local_dir0 * left_knee_local_dir1, dim=-1)
    left_leg_dot = torch.clamp(left_leg_dot, -1.0, 1.0)
    left_leg_theta = torch.acos(left_leg_dot)
    left_leg_theta = torch.where(left_knee_local_dir0[..., 1] >= 0, left_leg_theta, -left_leg_theta)
    left_leg_q = quat_from_angle_axis(left_leg_theta, left_knee_local_dir.unsqueeze(0))
    left_hip_rot = quat_mul(left_hip_rot, left_leg_q)
    

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., right_thigh_id, :] = right_hip_rot
    new_local_rotation[..., right_shin_id, :] = right_knee_q
    new_local_rotation[..., left_thigh_id, :] = left_hip_rot
    new_local_rotation[..., left_shin_id, :] = left_knee_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)
    
    return new_motion

def drop_nodes(motion, dropped_nodes: list):
    new_sk_state = motion.drop_nodes_by_names(dropped_nodes)
    if type(motion) is SkeletonMotion:
        return SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)
    return new_sk_state

def combine_joints(local_rotations: torch.Tensor, root_global_rotation: torch.Tensor, joints_combine: dict, joint_mapping: dict, dropped_nodes: list=[]):
    rotations_dict = {}
    for i in range(len(SMPL_MUJOCO_NAMES)):
        rotations_dict[SMPL_MUJOCO_NAMES[i]] = local_rotations[:, i]
    for target, sources in joints_combine.items():
        new_rotation = rotations_dict[sources[0]]
        for source in sources[1:]:
            new_rotation = quat_mul(new_rotation, rotations_dict[source])
        rotations_dict = {k: v for k, v in rotations_dict.items() if (k not in sources or k in target)}

    rotations_dict = {k: v for k, v in rotations_dict.items() if (k not in dropped_nodes)}
    rotations_dict = {joint_mapping[k]: v for k, v in rotations_dict.items()}
    print(rotations_dict.keys())

    rotations_dict["pelvis"] = root_global_rotation

    tensors = [rotations_dict[joint_name] for joint_name in body_names_humanoid28]
    new_rotations = torch.stack(tensors, dim=1)
    print(new_rotations.shape)
    return new_rotations

def retarget():
    # load retarget config
    retarget_data_path = "./retarget_interhuman_to_humanoid28.json"
    with open(retarget_data_path) as f:
        retarget_data = json.load(f)

    # load and visualize t-pose files
    source_tpose = SkeletonState.from_file(retarget_data["source_tpose"])
    # if VISUALIZE:
        # animate3D(source_tpose.global_translation.unsqueeze(0).repeat(2, 1, 1), q=source_tpose.global_rotation[0].unsqueeze(0).repeat(2, 1, 1), title="source tpose")
        # plot_tpose(source_tpose.global_translation)

    target_tpose = SkeletonState.from_file(retarget_data["target_tpose"])
    # if VISUALIZE:
    #     animate3D(target_tpose.global_translation.unsqueeze(0).repeat(2, 1, 1), q=target_tpose.global_rotation[0].unsqueeze(0).repeat(2, 1, 1), title="target tpose")
    #     plot_tpose(target_tpose.global_translation)

    # load and visualize source motion sequence
    source_motion = SkeletonMotion.from_file(retarget_data["source_motion"])
    if VISUALIZE:
        animate3D(source_motion.global_translation, q=source_motion.global_rotation[:, 0], title="source motion")

    # parse data from retarget config
    joint_mapping = retarget_data["joint_mapping"]
    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])

    # joints combine
    dropped_nodes = ["L_Thorax", "R_Thorax", "L_Wrist", "R_Wrist", "L_Toe", "R_Toe", "Spine", "Chest", "Neck"]
    # source_tpose = drop_nodes(source_tpose, dropped_nodes)
    # source_motion = drop_nodes(source_motion, dropped_nodes)
    new_rotations = combine_joints(source_motion.local_rotation, source_motion.global_rotation[:, 0], retarget_data["joints_combine"], retarget_data["joint_mapping"], dropped_nodes)
    _source_motion = SkeletonState.from_rotation_and_root_translation(target_tpose.skeleton_tree, new_rotations, source_motion.root_translation, is_local=True)
    _source_motion = SkeletonMotion.from_skeleton_state(_source_motion, fps=source_motion.fps)
    animate3D(_source_motion.global_translation, q=_source_motion.global_rotation[:, 0], title="after dropping nodes")
    target_motion = _source_motion

    # test: rename joints in source motion
    # # source_joint_names, target_joint_names = [item for item in SMPL_BONE_ORDER_NAMES if item not in dropped_nodes], [item for item in SMPL_MUJOCO_NAMES if item not in dropped_nodes]
    # # smpl_2_mujoco = [source_joint_names.index(q) for q in target_joint_names if q in source_joint_names]
    # local_rotations = source_motion.local_rotation.reshape(-1, len(source_joint_names), 4) #[..., smpl_2_mujoco, :].clone()
    # _source_motion = SkeletonState.from_rotation_and_root_translation(source_tpose.skeleton_tree, local_rotations, source_motion.root_translation, is_local=True)
    # _source_motion = SkeletonMotion.from_skeleton_state(_source_motion, fps=source_motion.fps)
    # _target_motion = _source_motion
    # animate3D(_source_motion.global_translation, q=_source_motion.global_rotation[:, 0], title="after renaming joints")

    # run retargeting
    # target_motion = source_motion.retarget_to_by_tpose(
    #   joint_mapping=retarget_data["joint_mapping"],
    #   source_tpose=source_tpose,
    #   target_tpose=target_tpose,
    #   rotation_to_target_skeleton=rotation_to_target_skeleton,
    #   scale_to_target_skeleton=retarget_data["scale"]
    # )
    # if VISUALIZE:
    #     animate3D(target_motion.global_translation, q=target_motion.global_rotation[:, 0], title="retargeted motion")

    # keep frames between [trim_frame_beg, trim_frame_end - 1]
    frame_beg = retarget_data["trim_frame_beg"]
    frame_end = retarget_data["trim_frame_end"]
    if (frame_beg == -1): frame_beg = 0
    if (frame_end == -1): frame_end = target_motion.local_rotation.shape[0]
        
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    local_rotation = local_rotation[frame_beg:frame_end, ...]
    root_translation = root_translation[frame_beg:frame_end, ...]
      
    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # need to convert some joints from 3D to 1D (e.g. elbows and knees)
    target_motion = project_joints(target_motion)

    # move the root so that the feet are on the ground
    local_rotation = target_motion.local_rotation
    root_translation = target_motion.root_translation
    tar_global_pos = target_motion.global_translation
    min_h = torch.min(tar_global_pos[..., 2])
    root_translation[:, 2] += -min_h
    
    # adjust the height of the root to avoid ground penetration
    root_height_offset = retarget_data["root_height_offset"]
    root_translation[:, 2] += root_height_offset
    
    new_sk_state = SkeletonState.from_rotation_and_root_translation(target_motion.skeleton_tree, local_rotation, root_translation, is_local=True)
    target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

    # save retargeted motion
    target_motion.to_file(retarget_data["target_motion_path"])

    # visualize retargeted motion
    if VISUALIZE:
        animate3D(target_motion.global_translation, q=target_motion.global_rotation[0], title="after projecting joints")
    
    return

if __name__ == '__main__':
    retarget()