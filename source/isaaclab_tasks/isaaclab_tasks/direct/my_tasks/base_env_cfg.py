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

# terrain
from .terrain.terrain_generator_cfg import ROUGH_TERRAINS_CFG
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

# motion directory
MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    # env
    observation_space: int = MISSING
    action_space: int = MISSING
    state_space: int = 0
    num_amp_observations: int = 2
    amp_observation_space: int = MISSING
    relative_pose_observation: int = 0

    # reward
    reward: list = ["zero"]
    
    # motions
    action_clip: list = MISSING
    init_root_height: float = 0.15
    early_termination: bool = True
    key_body_names: list = MISSING
    termination_bodies: list = MISSING
    termination_heights: list = MISSING
    reference_body: str = MISSING
    sync_motion: bool = False # apply reference actions instead of predicted actions to robots
    reset_strategy: str = "default"  # default, random, random-start (time zero from dataset)

    # two-character config
    require_relative_pose: bool = False # require precompute relative body positions between two robots
    require_another_pose: bool = False # require the local pose information of another character

    # simulation
    episode_length_s = -1 #10.0 # 10s * 30fps = 300 frames
    decimation = 2
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
    terrain: str = "default"
    scene: InteractiveSceneCfg = InteractiveSceneCfg(env_spacing=7.0, replicate_physics=True)
    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # visual_material=sim_utils.MdlFileCfg(
        #     mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        #     project_uvw=True,
        #     texture_scale=(0.25, 0.25),
        # ),
        visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        debug_vis=False,
    )

    # robot
    robot_format: str = MISSING

    # Create the markers configuration
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
    marker_green_small_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/GreenMarkers",
        markers={
            "marker": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )
    marker_red_small_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/GreenMarkers",
        markers={
            "marker": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        }
    )

class EnvCfg1Robot(BaseEnvCfg):
    robot1: ArticulationCfg = MISSING
    motion_file_1: str = MISSING

class EnvCfg2Robots(BaseEnvCfg):
    robot1: ArticulationCfg = MISSING
    robot2: ArticulationCfg = MISSING
    motion_file_1: str = MISSING
    motion_file_2: str = MISSING

class EnvCfg1RobotSMPL(EnvCfg1Robot):
    robot_format = "SMPL"
    robot1: ArticulationCfg = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot1")

    init_root_height = 0.25
    action_clip = [-0.06, 0.06]
    termination_bodies = ["Pelvis", "Head"]
    termination_heights = [0.5, 0.8]
    observation_space = 151 
    action_space = 69 
    amp_observation_space = observation_space
    key_body_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe", "Head"]
    reference_body = "Pelvis"

class EnvCfg2RobotsSMPL(EnvCfg2Robots):
    robot_format = "SMPL"
    robot1: ArticulationCfg = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot1")
    robot2: ArticulationCfg = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot2")

    init_root_height = 0.25
    action_clip = [-0.06, 0.06]
    termination_bodies = ["Pelvis", "Head"]
    termination_heights = [0.5, 0.8]
    observation_space = 151 * 2
    action_space = 69 * 2
    amp_observation_space = observation_space
    key_body_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe", "Head"]
    reference_body = "Pelvis"

class EnvCfg1RobotHumanoid28(EnvCfg1Robot):
    robot_format = "humanoid"
    robot1 = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot1").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )

    action_clip = [None, None]
    termination_bodies = ["torso", "head"]
    termination_heights = [0.4, 0.7]
    observation_space = 69
    action_space = 28
    amp_observation_space = observation_space
    key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
    reference_body = "torso"

class EnvCfg2RobotHumanoid28(EnvCfg2Robots):
    robot_format = "humanoid"
    robot1 = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot1").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )

    robot2 = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot2").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,
                damping=None,
            ),
        },
    )
    action_clip = [None, None]
    termination_bodies = ["torso", "head"]
    termination_heights = [0.4, 0.7]
    observation_space = 69 * 2
    action_space = 28 * 2
    amp_observation_space = observation_space
    key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
    reference_body = "torso"
