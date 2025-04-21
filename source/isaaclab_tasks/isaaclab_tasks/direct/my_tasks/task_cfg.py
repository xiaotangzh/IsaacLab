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
    reset_strategy = "random_start"
    sync_motion = False

    require_another_pose = True
    observation_space = 2 * 151 * 2
    amp_observation_space =  2 * 151

@configclass
class AmpInterHumanEnvCfg(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/26_1.npz")

    reward = ["ones"]
    reset_strategy = "random"

### PPO
@configclass
class PPOInterHumanEnvCfg(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/26_1.npz")

    reward = ["com acc"]
    reset_strategy = "default"

@configclass
class PPOInterHumanEnvCfg2Robots(EnvCfg2RobotsSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/1_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman/1_2.npz")

    reward = ["imitation"]
    reset_strategy = "random_start"

@configclass
class PPOHumanoidEnvCfg(EnvCfg1RobotHumanoid):
    motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid28/humanoid_walk.npz")
    reward = ["com_acc", "stand_forward"]
    reset_strategy = "default"

    terrain = "uneven"
    init_root_height = 2.0
    episode_length_s = 30.0

    scene = InteractiveSceneCfg(num_envs=16, env_spacing=2.0, replicate_physics=True)


### HRL
@configclass
class HRLInterHumanEnvCfg(EnvCfg1RobotSMPL):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/6929_1.npz")

    reward = ["com_acc"]
    reset_strategy = "random_start"

    terrain = "uneven"
    init_root_height = 0.3

@configclass
class HRLHumanoidEnvCfg(EnvCfg1RobotHumanoid):
    motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid28/humanoid_run.npz")

    reward = ["com_acc"]
    reset_strategy = "random_start"

    # terrain = "uneven"
    init_root_height = 0.15

    # sync_motion = True

    scene = InteractiveSceneCfg(num_envs=16, env_spacing=1.0, replicate_physics=True)