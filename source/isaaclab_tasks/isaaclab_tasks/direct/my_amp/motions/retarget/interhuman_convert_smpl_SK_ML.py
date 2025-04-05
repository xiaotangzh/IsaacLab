from convert_interhuman_isaac import run
from motion_lib import MotionLib
from motion_lib import animate3D
import torch
import argparse
import numpy as np
import os
from tqdm import tqdm

def convert(path, file_name, min_frames: int | None=None):
    SKMotion = run(
        in_file=f"in_files/{file_name}.pkl",
        SKMotion_out_file=f"{path}/{file_name}.npy"
    )
    
    with open(f"in_files/{file_name}.txt", 'r', encoding='utf-8') as text_file:
        text = text_file.read().splitlines()
    
    # check SkeletonMotion
    # animate3D(SKMotion.global_translation.reshape(-1, 24, 3))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    body_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    dof_names = ['L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'L_Knee_x', 'L_Knee_y', 'L_Knee_z', 'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 'L_Toe_x', 'L_Toe_y', 'L_Toe_z', 'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z', 'R_Ankle_x', 'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z', 'Torso_x', 'Torso_y', 'Torso_z', 'Spine_x', 'Spine_y', 'Spine_z', 'Chest_x', 'Chest_y', 'Chest_z', 'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z', 'L_Thorax_x', 'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', 'L_Hand_x', 'L_Hand_y', 'L_Hand_z', 'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', 'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 'R_Wrist_z', 'R_Hand_x', 'R_Hand_y', 'R_Hand_z']
    
    dof_body_ids = [i for i in range(1,24)]
    dof_offsets = []
    previous_dof_name = "null"
    for dof_offset, dof_name in enumerate(dof_names):
        if dof_name[:-2] != previous_dof_name:  # remove the "_x/y/z"
            previous_dof_name = dof_name[:-2]
            dof_offsets.append(dof_offset)
    dof_offsets.append(len(dof_names))

    motion_lib = MotionLib(
        motion_file=SKMotion,
        dof_body_ids=dof_body_ids,
        dof_offsets=dof_offsets,
        device=torch.device('cpu')
    )
    
    data = motion_lib.get_motion_state()
    data['dof_names'] = dof_names
    data['body_names'] = body_names
    data['text'] = text
    
    w_last = False
    if w_last is False:
        data['root_rotation'] = fromXYZWtoWXYZ(data['root_rotation'])
        data['local_rotations'] = fromXYZWtoWXYZ(data['local_rotations'])
        data['body_rotations'] = fromXYZWtoWXYZ(data['body_rotations'])
    # print(data.keys())
    # print(data['dof_positions'].shape)
    animate3D(data['body_positions'], highlight_joint=0, q=data['body_rotations'][:,0], w_last=w_last)
    # animate3D(data['dof_positions'].reshape(-1, 24, 3))
    
    if min_frames is None or (min_frames is not None and data['dof_positions'].shape[0] > min_frames):
        np.savez(f"{path}/{file_name}.npz", **data)
        
def fromXYZWtoWXYZ(quats: torch.Tensor):
    w = quats[..., 3:] 
    xyz = quats[..., :3] 
    return torch.cat([w, xyz], dim=-1) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="in_files/1.pkl")
    args = parser.parse_args()
    
    min_frames = 300
    
    if 'pkl' not in args.in_file:
        file_names = [f for f in os.listdir(args.in_file) if os.path.isfile(os.path.join(args.in_file, f))]
        for file_name in tqdm(file_names):
            convert(path="../InterHuman", file_name=file_name.replace('.pkl',''), min_frames=min_frames)
    elif 'pkl' in args.in_file:
        convert(path="../InterHuman", file_name=args.in_file.replace('.pkl','').replace('in_files/',''), min_frames=min_frames)
    else:
        raise ValueError("Invalid input file format. Please provide a .pkl file.")