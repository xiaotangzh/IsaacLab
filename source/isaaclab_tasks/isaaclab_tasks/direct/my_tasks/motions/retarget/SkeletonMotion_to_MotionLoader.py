import torch
from motion_lib import MotionLib
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from visualization import animate3D
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

body_names_SMPL = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
dof_names_SMPL = ['L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'L_Knee_x', 'L_Knee_y', 'L_Knee_z', 'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 'L_Toe_x', 'L_Toe_y', 'L_Toe_z', 'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z', 'R_Ankle_x', 'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z', 'Torso_x', 'Torso_y', 'Torso_z', 'Spine_x', 'Spine_y', 'Spine_z', 'Chest_x', 'Chest_y', 'Chest_z', 'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z', 'L_Thorax_x', 'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', 'L_Hand_x', 'L_Hand_y', 'L_Hand_z', 'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', 'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 'R_Wrist_z', 'R_Hand_x', 'R_Hand_y', 'R_Hand_z']
body_offsets_SMPL = [i for i in range(1, 24)]
dof_offsets_SMPL = [i for i in range(0, 24*3, 3)]

body_names_humanoid28 = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'left_upper_arm', 'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']
dof_names_humanoid28 = ['abdomen_x', 'abdomen_y', 'abdomen_z', 'neck_x', 'neck_y', 'neck_z', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_elbow', 'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'left_elbow', 'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z']
body_offsets_humanoid28 = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
dof_offsets_humanoid28 = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]

def get_names(skeleton):
    match skeleton:
        case "humanoid28":
            return (body_names_humanoid28, dof_names_humanoid28, body_offsets_humanoid28, dof_offsets_humanoid28)
        case "SMPL":
            return (body_names_SMPL, dof_names_SMPL, body_offsets_SMPL, dof_offsets_SMPL)

def SkeletonMotion_to_MotionLoader(path, skeleton: str="humanoid28", visualize=False, w_last=False):
    body_names, dof_names, body_offsets, dof_offsets = get_names(skeleton)

    SKMotion = SkeletonMotion.from_file(path)
    motion_lib = MotionLib(
        motion_file=SKMotion,
        dof_body_ids=body_offsets,
        dof_offsets=dof_offsets,
        device=torch.device('cpu')
    )

    data = motion_lib.get_motion_state()
    data['dof_names'] = dof_names
    data['body_names'] = body_names

    if not w_last:
        data['root_rotation'] = fromXYZWtoWXYZ(data['root_rotation'])
        data['local_rotations'] = fromXYZWtoWXYZ(data['local_rotations'])
        data['body_rotations'] = fromXYZWtoWXYZ(data['body_rotations'])
    # print_dict(data)
    # print(data['dof_positions'].shape)

    if visualize: animate3D(data['body_positions'], highlight_joint=0, q=data['body_rotations'][:,0], w_last=w_last)
    return data

def fromXYZWtoWXYZ(quats: torch.Tensor):
    w = quats[..., 3:] 
    xyz = quats[..., :3] 
    return torch.cat([w, xyz], dim=-1) 
