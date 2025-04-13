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

# marker
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
import isaaclab.sim as sim_utils

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class EnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 30.0 # 10s * 30fps = 300 frames
    decimation = 2

    # num_persons = 2
    # observation_space = 151 * num_persons
    # action_space = 69 * num_persons
    # state_space = 0
    # num_amp_observations = 2
    # amp_observation_space = observation_space

    early_termination = True
    termination_bodies = ["Pelvis", "Head"]
    termination_heights = [0.5, 0.8]
    action_clip = [-0.1, 0.1]
    
    # reward
    reward: list = []
    
    # motions
    # motion_file_1: str = MISSING
    # motion_file_2: str = MISSING
    reference_body = "Pelvis"
    sync_motion = False # apply reference actions instead of predicted actions to robots
    reset_strategy: str = MISSING  # default, random, random-start
    
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    dt = 1 / 60
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=5.0, replicate_physics=True)

    # robot
    # robot1: ArticulationCfg = MISSING 
    # robot2: ArticulationCfg = MISSING 

    # Create the markers configuration
    # This creates two marker prototypes, "marker1" and "marker2" which are spheres with a radius of 1.0.
    # The color of "marker1" is red and the color of "marker2" is green.
    marker_green_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/GreenMarkers",
        markers={
            "marker": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )
    marker_red_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/RedMarkers",
        markers={
            "marker": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )
    # Create the markers instance
    # This will create a UsdGeom.PointInstancer prim at the given path along with the marker prototypes.
    # marker = VisualizationMarkers(marker_cfg)

class EnvCfg1Robot(EnvCfg):
    num_persons = 1
    observation_space = 151 * num_persons
    action_space = 69 * num_persons
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = observation_space

    # robot
    robot1 = None

    motion_file_1: str = MISSING

class EnvCfg2Robots(EnvCfg):
    num_persons = 2
    observation_space = 151 * num_persons
    action_space = 69 * num_persons
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = observation_space

    # robot
    robot1: ArticulationCfg = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot1")
    robot2: ArticulationCfg = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot2")

    motion_file_1: str = MISSING
    motion_file_2: str = MISSING

@configclass
class AmpInterHumanEnvCfg2Robots(EnvCfg2Robots):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/1_1.npz")
    motion_file_2 = os.path.join(MOTIONS_DIR, "InterHuman/1_2.npz")
    # robot1 = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot1")
    # robot2 = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot2")

    reward = ["ones"]
    reset_strategy = "random"

@configclass
class AmpInterHumanEnvCfg(EnvCfg1Robot):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/26_1.npz")
    robot1 = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot1")

    reward = ["ones"]
    reset_strategy = "random"
    
@configclass
class PPOEnvCfg(EnvCfg1Robot):
    motion_file_1 = os.path.join(MOTIONS_DIR, "InterHuman/26_1.npz")
    robot1 = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot1")

    reward = ["com acc"]
    reset_strategy = "default"

@configclass
class PPOHumanoidEnvCfg(EnvCfg1Robot):
    motion_file_1 = os.path.join(MOTIONS_DIR, "humanoid/humanoid_walk.npz")
    robot1 = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )

    reward = ["com acc"]
    reset_strategy = "default"

    termination_bodies = ["torso", "head"]
    termination_heights = [0.5, 0.8]
    reference_body = "torso"

    observation_space = 69
    action_space = 28
    amp_observation_space = observation_space

    action_clip = [None, None]