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

import numpy as np
import os
import yaml

from poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.core.rotation3d import *
# from isaacgym.torch_utils import *

import torch_utils
import rotations

import torch

class MotionLib():
    def __init__(self, motion_file, dof_body_ids, dof_offsets, device):
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._device = device
        self._load_motions(motion_file)

        motions = self._motions
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float()
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float()
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float()
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float()
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float()
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float()

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        return

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]
    
    def load_external_motion(self, motions):
        self.gts = motions["global_translation"]
        self.grs = motions["global_rotation"]
        self.lrs = motions["local_rotation"]
        self.grvs = motions["global_root_velocity"]
        self.gravs = motions["global_root_angular_velocity"]
        self.dvs = motions["dof_vels"]

    def get_motion_state(self, start: int=0, end: int | None=None):
        if end is None: end = self.gts.shape[0]
        
        root_pos = self.gts[start:end, 0]
        root_rot = self.grs[start:end, 0]
        root_vel = self.grvs[start:end]
        root_ang_vel = self.gravs[start:end]
        
        dof_pos = self._local_rotation_to_dof(self.lrs[start:end], 'exp_map')
        dof_vel = self.dvs[start:end]
        
        local_rot = self.lrs[start:end]
        
        rigid_body_pos = self.gts[start:end]
        rigid_body_rot = self.grs[start:end]
        
        data = {
            'fps': self._motion_fps[0],
            'root_translation': root_pos,
            'root_rotation': root_rot,
            'root_linear_velocity': root_vel,
            'root_angular_velocity': root_ang_vel,
            'dof_positions': dof_pos,
            'dof_velocities': dof_vel,
            'local_rotations': local_rot,
            'body_positions': rigid_body_pos,
            'body_rotations': rigid_body_rot,
        }

        return data
    
    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []

        total_len = 0.0

        if type(motion_file) is not SkeletonMotion:
            motion_files, motion_weights = self._fetch_motion_files(motion_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            if type(motion_file) is not SkeletonMotion:
                print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
                curr_motion = SkeletonMotion.from_file(curr_file)
            else:
                curr_motion = motion_file

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
 
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            
            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)


        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        return

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            motion_weights = []

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = motion_config['motions']
            for motion_entry in motion_list:
                curr_file = motion_entry['file']
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file)
                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def _compute_motion_dof_vels(self, motion: SkeletonMotion):
        num_frames = motion.global_translation.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            dof_vels.append(frame_dof_vel)

        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels
    
    def _compute_motion_dof_vels_external(self, motion, fps):
        num_frames = motion["global_translation"].shape[0]
        dt = 1.0 / fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion["local_rotation"][f]
            local_rot1 = motion["local_rotation"][f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            dof_vels.append(frame_dof_vel)

        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels

    # jp hack
    # get rid of this ASAP, need a proper way of projecting from max coords to reduced coords
    def _local_rotation_to_dof(self, local_rot, joint_3d_format):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_q = local_rot[:, body_id]
                if joint_3d_format == "exp_map":
                    formatted_joint = torch_utils.quat_to_exp_map(joint_q, w_last=True)
                elif joint_3d_format == "xyz":
                    x, y, z = rotations.get_euler_xyz(joint_q, w_last=True)
                    formatted_joint = torch.stack([x, y, z], dim=-1)
                else:
                    raise ValueError(f"Unknown 3d format '{joint_3d_format}'")

                dof_pos[:, joint_offset : (joint_offset + joint_size)] = formatted_joint
            elif joint_size == 1:
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(
                    joint_q, w_last=True
                )
                joint_theta = (
                    joint_theta * joint_axis[..., 1]
                )  # assume joint is always along y axis

                joint_theta = rotations.normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert False

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        dof_vel = torch.zeros([self._num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset : (joint_offset + joint_size)] = joint_vel

            elif joint_size == 1:
                assert joint_size == 1
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[
                    1
                ]  # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert False

        return dof_vel

if __name__ == "__main__":
    dof_body_ids = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 ]
    dof_offsets = range(24*3+1) # todo
    key_body_ids = [4, 8, -6, -1] #todo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    body_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    dof_names = ['Pelvis_x', 'Pelvis_y', 'Pelvis_z', 'L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'L_Knee_x', 'L_Knee_y', 'L_Knee_z', 'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 'L_Toe_x', 'L_Toe_y', 'L_Toe_z', 'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z', 'R_Ankle_x', 'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z', 'Torso_x', 'Torso_y', 'Torso_z', 'Spine_x', 'Spine_y', 'Spine_z', 'Chest_x', 'Chest_y', 'Chest_z', 'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z', 'L_Thorax_x', 'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', 'L_Hand_x', 'L_Hand_y', 'L_Hand_z', 'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', 'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 'R_Wrist_z', 'R_Hand_x', 'R_Hand_y', 'R_Hand_z']
    
    motion_lib = MotionLib(
        motion_file="../InterHuman/1.npy",
        dof_body_ids=dof_body_ids,
        dof_offsets=dof_offsets,
        key_body_ids=key_body_ids,
        device=torch.device('cpu')
    )
    
    data = motion_lib.get_motion_state(1, 1)
    data['dof_names'] = dof_names
    data['body_names'] = body_names
    print(data.keys())
    print(data['dof_positions'].shape)
    np.savez("../InterHuman/1.npz", **data)