# This code is adapted from https://github.com/zhengyiluo/phc/ and generalized to work with any humanoid.
# https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/convert_amass_isaac.py

from ast import Try
import torch
# import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as sRot
import glob
import os
import sys
import pdb
import os.path as osp
from pathlib import Path

sys.path.append(os.getcwd())

# from smpl_sim.khrylib.utils import get_body_qposaddr
# from smpl_sim.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import scipy.ndimage.filters as filters
from typing import List, Optional
from tqdm import tqdm
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
import argparse
import pickle

def run(in_file: str, out_file: str, SKMotion_out_file = None):

    robot_cfg = {
        "mesh": False,
        "model": "smpl",
        "upright_start": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    # robot_cfg = {
    #     "mesh": False,
    #     "rel_joint_lm": True,
    #     "upright_start": True,
    #     "remove_toe": False,
    #     "real_weight": True,
    #     "real_weight_porpotion_capsules": True,
    #     "real_weight_porpotion_boxes": True,
    #     "replace_feet": True,
    #     "masterfoot": False,
    #     "big_ankle": True,
    #     "freeze_hand": False,
    #     "box_body": False,
    #     "master_range": 50,
    #     "body_params": {},
    #     "joint_params": {},
    #     "geom_params": {},
    #     "actuator_params": {},
    #     "model": "smpl",
    #     "sim": "isaacgym",
    # }
    print(robot_cfg)

    smpl_local_robot = LocalRobot(
        robot_cfg,
        data_dir="./smpl",
    )

    # amass_data = joblib.load(in_file)
    with open(in_file, 'rb') as file:
        amass_data = pickle.load(file)

    double = False

    # mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    mujoco_joint_names = SMPL_MUJOCO_NAMES
    joint_names = SMPL_BONE_ORDER_NAMES
    

    amass_full_motion_dict = {}
    interhuman_keys = ["person1", "person2"]
    for key_name in tqdm(interhuman_keys):
        smpl_data_entry = amass_data[key_name]
        B = smpl_data_entry['pose_body'].shape[0]

        pose_aa = smpl_data_entry['pose_body'].copy()
        root_orient = smpl_data_entry['root_orient'].copy()
        root_trans = smpl_data_entry['trans'].copy()
        B = pose_aa.shape[0]
        
        pose_aa = np.concatenate([root_orient, pose_aa], axis=1)

        beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy()
        if len(beta.shape) == 2:
            beta = beta[0]

        gender = smpl_data_entry.get("gender", "neutral")
        fps = 30.0

        if isinstance(gender, np.ndarray):
            gender = gender.item()

        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        if gender == "neutral":
            gender_number = [0]
        elif gender == "male":
            gender_number = [1]
        elif gender == "female":
            gender_number = [2]
        else:
            import ipdb
            ipdb.set_trace()
            raise Exception("Gender Not Supported!!")

        smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]
        batch_size = pose_aa.shape[0]
        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1) # 23*3 joints + zero values L/R hands = 24 joints
        pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()

        num = 1
        if double:
            num = 2
        for idx in range(num):
            pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)

            gender_number, beta[:], gender = [0], 0, "neutral"
            print("using neutral model")

            smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
            smpl_local_robot.write_xml("../assets/smpl_humanoid_1.xml")
            skeleton_tree = SkeletonTree.from_mjcf("../assets/smpl_humanoid_1.xml")

            root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                torch.from_numpy(pose_quat),
                root_trans_offset,
                is_local=True)
            
            

            if robot_cfg['upright_start']:
                pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)  # should fix pose_quat as well here...

                new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
                pose_quat = new_sk_state.local_rotation.numpy()

                ############################################################
                # key_name_dump = key_name + f"_{idx}"
                key_name_dump = key_name
                if idx == 1:
                    left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]
                    pose_quat_global = pose_quat_global[:, left_to_right_index]
                    pose_quat_global[..., 0] *= -1
                    pose_quat_global[..., 2] *= -1

                    root_trans_offset[..., 1] *= -1
                ############################################################
            
            # save as SkeletonMotion
            if SKMotion_out_file:
                target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=int(fps))
                # target_motion.to_file(SKMotion_out_file)
                # draw3D(target_motion.global_transformation[100,:,-3:])
                # animate3D(target_motion.global_transformation[:,:,-3:])
                # animate3D(target_motion.global_translation)
                # print(target_motion.global_transformation.shape)
                return target_motion

            new_motion_out = {}
            new_motion_out['pose_quat_global'] = pose_quat_global
            new_motion_out['pose_quat'] = pose_quat
            new_motion_out['trans_orig'] = root_trans
            new_motion_out['root_trans_offset'] = root_trans_offset
            new_motion_out['beta'] = beta
            new_motion_out['gender'] = gender
            new_motion_out['pose_aa'] = pose_aa
            new_motion_out['fps'] = fps
            amass_full_motion_dict[key_name_dump] = new_motion_out

    Path(out_file).parents[0].mkdir(parents=True, exist_ok=True)
    # joblib.dump(amass_full_motion_dict, out_file)
    with open(out_file, "wb") as f:
        pickle.dump(amass_full_motion_dict, f)

    return 

def draw3D(tensor):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = tensor[:, 0]
    y = tensor[:, 1]
    z = tensor[:, 2]

    ax.scatter(x, y, z, c='r', marker='o', s=50)

    plt.show()
    
def animate3D(tensor):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = ax.set_title('Frame: 0')

    scatter = ax.scatter([], [], [], c='red', s=50)

    def update(frame):
        x = tensor[frame, :, 0]
        y = tensor[frame, :, 1]
        z = tensor[frame, :, 2]
        scatter._offsets3d = (x, y, z)
        title.set_text(f'Frame: {frame}/400')
        return scatter, title

    ani = FuncAnimation(
        fig, update, frames=tensor.shape[0],
        interval=50, blit=False
    )

    plt.show()  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="in_files/1.pkl")
    parser.add_argument("--out_file", type=str, default="out_files/1.pkl")
    parser.add_argument("--SKMotion_out_file", type=str, default="out_files/1.npy")
    args = parser.parse_args()
    run(
        in_file=args.in_file,
        out_file=args.out_file,
        SKMotion_out_file=args.SKMotion_out_file,
    )
