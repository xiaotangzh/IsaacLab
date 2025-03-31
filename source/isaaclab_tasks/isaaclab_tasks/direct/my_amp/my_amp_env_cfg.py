# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import HUMANOID_28_CFG
from isaaclab_assets.robots.smpl import SMPL_CFG
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class MyAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces
    observation_space = 42
    action_space = 69
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = observation_space

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "Pelvis"
    reset_strategy = "random-start"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """
    
    sync_motion = True # apply reference actions instead of predicted actions to robots

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    # robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
    #     actuators={
    #         "body": ImplicitActuatorCfg(
    #             joint_names_expr=[".*"],
    #             velocity_limit=100.0,
    #             stiffness=None,
    #             damping=None,
    #         ),
    #     },
    # )
    robot: ArticulationCfg = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class MyAmpInterHumanEnvCfg(MyAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "InterHuman/1.npz")

# @configclass
# class HumanoidAmpDanceEnvCfg(MyAmpEnvCfg):
#     motion_file = os.path.join(MOTIONS_DIR, "humanoid/humanoid_dance.npz")


# @configclass
# class HumanoidAmpRunEnvCfg(MyAmpEnvCfg):
#     motion_file = os.path.join(MOTIONS_DIR, "humanoid/humanoid_run.npz")


# @configclass
# class HumanoidAmpWalkEnvCfg(MyAmpEnvCfg):
#     motion_file = os.path.join(MOTIONS_DIR, "humanoid/humanoid_walk.npz")
