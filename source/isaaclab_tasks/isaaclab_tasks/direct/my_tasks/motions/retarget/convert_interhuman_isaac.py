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

def run(in_file: str, SKMotion_out_file = None, person: str = "person1"):

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

    smpl_local_robot = LocalRobot(
        robot_cfg,
        data_dir="./smpl",
    )

    # amass_data = joblib.load(in_file)
    with open(in_file, 'rb') as file:
        amass_data = pickle.load(file)

    double = False

    # mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    target_joint_names = SMPL_MUJOCO_NAMES
    source_joint_names = SMPL_BONE_ORDER_NAMES


    # start retargeting
    key_name = person # ["person1", "person2"]
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

    smpl_2_mujoco = [source_joint_names.index(q) for q in target_joint_names if q in source_joint_names]
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
        smpl_local_robot.write_xml("../assets/smpl_humanoid.xml")
        skeleton_tree = SkeletonTree.from_mjcf("../assets/smpl_humanoid.xml")
        #TODO:
        # skeleton_tree = SkeletonTree.from_mjcf("../assets/amp_humanoid.xml")
        # print(skeleton_tree.node_names, skeleton_tree.num_joints)
        # print(pose_quat.shape)

        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)
        
        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)  # should fix pose_quat as well here...

            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
        
        # save as SkeletonMotion
        if SKMotion_out_file:
            target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=int(fps))
            return target_motion

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
