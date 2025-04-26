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
class Amp_InterHuman_2Robots(EnvCfg2RobotsSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/1_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman/1_2.npz")

    reset_strategy = "random_start"
    sync_motion = False

    # require_another_pose = True
    observation_space = 2 * 151 
    amp_observation_space =  2 * 151

@configclass
class Amp_InterHuman(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/26_1.npz")

    reset_strategy = "random"
    sync_motion = False

    scene = InteractiveSceneCfg(env_spacing=5.0, replicate_physics=True)

@configclass
class AMP_Humanoid(EnvCfg1RobotHumanoid28):
    motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid28/humanoid_run.npz")

    reset_strategy = "random"
    sync_motion = False

    scene = InteractiveSceneCfg(env_spacing=5.0, replicate_physics=True)

### PPO
@configclass
class PPO_InterHuman(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/1_1.npz")

    reward = ["imitation"]
    scene = InteractiveSceneCfg(env_spacing=3.0, replicate_physics=True)
    reset_strategy = "default"
    sync_motion = True

@configclass
class PPO_InterHuman_2Robots(EnvCfg2RobotsSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/1_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman/1_2.npz")

    sync_motion = False
    require_another_pose = True
    observation_space = 2 * 151 * 2
    amp_observation_space =  2 * 151
    scene = InteractiveSceneCfg(env_spacing=5.0, replicate_physics=True)
    reward = ["imitation"]
    reset_strategy = "random_start"

@configclass
class PPO_Humanoid(EnvCfg1RobotHumanoid28):
    motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid28/humanoid_walk.npz")
    reward = ["com_acc", "imitation"]
    reset_strategy = "random_start"
    sync_motion = False

    # terrain = "uneven"
    # init_root_height = 2.0
    episode_length_s = 30.0

    scene = InteractiveSceneCfg(env_spacing=5.0, replicate_physics=True)


### HRL
@configclass
class HRL_InterHuman(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/6929_1.npz")

    reward = ["com_acc"]
    reset_strategy = "random_start"

    terrain = "uneven"
    init_root_height = 0.3

@configclass
class HRL_Humanoid(EnvCfg1RobotHumanoid28):
    motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid28/humanoid_walk.npz")

    reward = ["stand_forward"]
    reset_strategy = "random_start"

    terrain = "uneven"
    # init_root_height = 0.15

    # sync_motion = True
    scene = InteractiveSceneCfg(env_spacing=2.5, replicate_physics=True)