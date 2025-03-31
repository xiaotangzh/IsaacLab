from convert_interhuman_isaac import run
from motion_lib import MotionLib
from motion_lib import animate3D
import torch
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="in_files/1.pkl")
    parser.add_argument("--out_file", type=str, default="out_files/1.pkl")
    args = parser.parse_args()
    SKMotion = run(
        in_file=args.in_file,
        out_file=args.out_file,
        SKMotion_out_file="../InterHuman/1.npy"
    )
    
    dof_body_ids = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 ]
    dof_offsets = range(24*3+1) # todo
    key_body_ids = [4, 8, -6, -1] #todo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    body_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    dof_names = ['Pelvis_x', 'Pelvis_y', 'Pelvis_z', 'L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'L_Knee_x', 'L_Knee_y', 'L_Knee_z', 'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 'L_Toe_x', 'L_Toe_y', 'L_Toe_z', 'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z', 'R_Ankle_x', 'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z', 'Torso_x', 'Torso_y', 'Torso_z', 'Spine_x', 'Spine_y', 'Spine_z', 'Chest_x', 'Chest_y', 'Chest_z', 'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z', 'L_Thorax_x', 'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', 'L_Hand_x', 'L_Hand_y', 'L_Hand_z', 'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', 'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 'R_Wrist_z', 'R_Hand_x', 'R_Hand_y', 'R_Hand_z']
    
    motion_lib = MotionLib(
        motion_file=SKMotion,
        dof_body_ids=dof_body_ids,
        dof_offsets=dof_offsets,
        key_body_ids=key_body_ids,
        device=torch.device('cpu')
    )
    
    data = motion_lib.get_motion_state()
    data['dof_names'] = dof_names
    data['body_names'] = body_names
    print(data.keys())
    print(data['dof_positions'].shape)
    animate3D(data['body_positions'])
    np.savez("../InterHuman/1.npz", **data)