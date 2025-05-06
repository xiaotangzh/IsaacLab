import os
from dataclasses import MISSING
from isaaclab_assets import HUMANOID_28_CFG
from .robots.smpl import SMPL_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# base configuration
from isaaclab_tasks.direct.my_tasks.base_env_cfg import *

# motion directory
MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

### AMP
@configclass
class AMP_InterHuman_2Robots(EnvCfg2RobotsSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/26_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/26_2.npz")

    observation_space = 151
    action_space = 69 
    amp_observation_space = observation_space

    reset_strategy = "random"
    sync_motion = False

@configclass
class AMP_InterHuman(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/1_1.npz")
    reset_strategy = "random_start"
    sync_motion = False
    reward = ["energy_penalty"]

@configclass
class AMP_Humanoid(EnvCfg1RobotHumanoid28):
    motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid28/humanoid_walk.npz")
    reset_strategy = "random_start"
    sync_motion = False
    # terrain = "rough"

@configclass
class AMP_Humanoid_rough_walk(AMP_Humanoid):
    motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid28/humanoid_walk.npz")
    reset_strategy = "random_start"
    sync_motion = False
    terrain = "rough"
    scene: InteractiveSceneCfg = InteractiveSceneCfg(env_spacing=2.5, replicate_physics=True)

### AIP
@configclass
class AIP_InterHuman_TestRobot(EnvCfg2RobotsSMPL):
    # sync motion to test_robot
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/26_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/26_2.npz")

    robot2 = None
    test_robot: ArticulationCfg = SMPL_CFG.replace(prim_path="/World/envs/env_.*/test_robot")

    reset_strategy = "random"
    sync_motion = "test_robot"

    key_body_num = 10
    interaction_space = key_body_num * key_body_num #* 3 * 2
    observation_space = 151 + interaction_space
    action_space = 69
    amp_observation_space =  151
    amp_inter_observation_space = interaction_space 
    interaction_modeling = True
    # key_body_names = ["L_Hand", "R_Hand", "Head", "L_Thorax", "R_Thorax"] # rel_pos + rel_vel
    # key_body_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe", "Head" , "L_Shoulder", "R_Shoulder", "L_Hip", "R_Hip", "Torso"]
    key_body_names = ["Head", "Torso", "L_Thorax", "R_Thorax", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Hand", "R_Hand"]
    # reward = ["energy_penalty"]
    init_root_height = 0.0
    action_clip = 0.1 #test:
    require_sync_frame_index = True


# SMPL body_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
@configclass
class AIP_InterHuman_2Robots(EnvCfg2RobotsSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/26_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/26_2.npz")

    reset_strategy = "random"

    # pjd1,4
    # key_body_num = 10
    # key_body_names = ["Head", "Torso", "L_Thorax", "R_Thorax", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Hand", "R_Hand"]
    # interaction_space = key_body_num * key_body_num #+ key_body_num * 3

    # pjd2
    key_body_num = 5
    key_body_names = ["Head", "L_Thorax", "R_Thorax", "L_Hand", "R_Hand"]
    interaction_space = key_body_num * key_body_num * 3 


    observation_space = 151 + interaction_space
    action_space = 69
    amp_observation_space =  151
    amp_inter_observation_space = interaction_space 
    interaction_modeling = True
    # reward = ["energy_penalty"]
    init_root_height = 0.0
    action_clip = 0.1 #test:
    require_sync_frame_index = False

### PPO
@configclass
class PPO_InterHuman(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/1_1.npz")

    reward = ["imitation"]
    reset_strategy = "default"
    sync_motion = True

@configclass
class PPO_InterHuman_2Robots(EnvCfg2RobotsSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/1_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/1_2.npz")

    sync_motion = False
    require_another_pose = True
    observation_space = 2 * 151 * 2
    amp_observation_space =  2 * 151
    reward = ["imitation"]
    reset_strategy = "random_start"

@configclass
class PPO_Humanoid(EnvCfg1RobotHumanoid28):
    # motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid28/humanoid_walk.npz")
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman_humanoid28/1_person1.npz")
    reward = ["com_acc", "imitation"]
    reset_strategy = "random_start"
    sync_motion = True

    # terrain = "rough"
    # init_root_height = 2.0
    episode_length_s = 30.0


### HRL
@configclass
class HRL_InterHuman(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman_SMPL/6929_1.npz")

    reward = ["com_acc"]
    reset_strategy = "random_start"

    terrain = "rough"
    init_root_height = 0.3

@configclass
class HRL_Humanoid(EnvCfg1RobotHumanoid28):
    motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid28/humanoid_walk.npz")

    # reward = ["com_acc"] 
    reset_strategy = "random_start"

    terrain = "rough"
    # init_root_height = 0.15

    # sync_motion = True