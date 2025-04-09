# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import HUMANOID_28_CFG
from .robots.smpl import SMPL_CFG
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class Env2RobotsCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 30.0 # 10s * 30fps = 300 frames
    decimation = 2

    # spaces
    num_persons = 2
    observation_space = 151 * num_persons
    action_space = 69 * num_persons
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = observation_space

    early_termination = True
    termination_bodies = ["Pelvis", "Head"]
    termination_heights = [0.5, 0.8]
    action_clip = [None, None] #[-0.1, 0.1]
    
    # reward
    reward = "ones"
    
    # motions
    motion_file_1: str = MISSING
    motion_file_2: str = MISSING
    reference_body = "Pelvis"
    sync_motion = False # apply reference actions instead of predicted actions to robots
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=5.0, replicate_physics=True)

    # robot
    robot1: ArticulationCfg = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot1")
    robot2: ArticulationCfg = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot2")

@configclass
class AmpInterHumanEnvCfg(Env2RobotsCfg):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/26_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman/26_2.npz")
    
@configclass
class PPOInterHumanEnvCfg(Env2RobotsCfg):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/26_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman/26_2.npz")
    