import os
from dataclasses import MISSING
from isaaclab_assets import HUMANOID_28_CFG
from .robots.smpl import SMPL_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# base configuration
from isaaclab_tasks.direct.my_tasks.base_cfg import *

# motion directory
MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

### AMP
@configclass
class AmpInterHumanEnvCfg2Robots(EnvCfg2RobotsSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/1_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman/1_2.npz")

    reward = ["ones"]
    reset_strategy = "random"

    require_relative_pose = True
    relative_pose_observation = 24 * 3
    observation_space = 2 * (151 + relative_pose_observation)
    amp_observation_space = observation_space

@configclass
class AmpInterHumanEnvCfg(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/26_1.npz")

    reward = ["ones"]
    reset_strategy = "random"
    

### PPO
@configclass
class PPOEnvCfg(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/26_1.npz")
    robot1 = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot1")

    reward = ["com acc"]
    reset_strategy = "default"

@configclass
class PPOHumanoidEnvCfg(EnvCfg1RobotHumanoid):
    motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid/humanoid_walk.npz")
    reward = ["com_acc", "stand_forward"]
    reset_strategy = "default"

    # action_clip = [None, None]
    terrain = "uneven"

    init_root_height = 2.0
    episode_length_s = 30.0

    scene = InteractiveSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=True)