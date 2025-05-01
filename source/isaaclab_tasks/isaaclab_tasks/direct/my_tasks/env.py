# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
from matplotlib.pyplot import isinteractive
import numpy as np
import torch
from isaaclab.envs.mdp.observations import projected_gravity
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab_tasks.direct.my_tasks.utils.env_utils import *
from isaaclab_tasks.direct.my_tasks.utils.utils import *
from isaaclab_tasks.direct.my_tasks.utils.rewards import *
from isaaclab_tasks.direct.my_tasks.utils.reward_utils import *
from isaaclab_tasks.direct.my_tasks.task_env_cfg import BaseEnvCfg
from isaaclab_tasks.direct.my_tasks.motions.motion_loader import MotionLoader, MotionLoaderHumanoid28, MotionLoaderSMPL
import sys
import random
import math

# marker
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
import isaaclab.sim as sim_utils

# terrain
from isaaclab.terrains import TerrainImporter

class Env(DirectRLEnv):
    cfg: BaseEnvCfg

    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.timestep = 0

        # action offset and scale
        dof_lower_limits = self.robot1.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot1.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        self.termination_heights = torch.tensor(self.cfg.termination_heights, device=self.device)

        # load motion
        if self.cfg.robot_format == "humanoid": MotionLoader = MotionLoaderHumanoid28
        else: MotionLoader = MotionLoaderSMPL
        self.motion_loader_1 = MotionLoader(motion_file=self.cfg.motion_file_1, device=self.device)
        self.motion_loader_2 = MotionLoader(motion_file=self.cfg.motion_file_2, device=self.device) if self.cfg.motion_file_2 is not None else None 
        self.sample_times = None # synchronize sampling times for two robots
        if self.cfg.episode_length_s < 0 or self.cfg.episode_length_s > self.motion_loader_1.duration:
            self.cfg.episode_length_s = self.motion_loader_1.duration

        # DOF and key body indexes
        key_body_names = self.cfg.key_body_names
        self.ref_body_index = self.robot1.data.body_names.index(self.cfg.reference_body)
        self.early_termination_body_indexes = [self.robot1.data.body_names.index(name) for name in self.cfg.termination_bodies]
        self.key_body_indexes = [self.robot1.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self.motion_loader_1.get_dof_index(self.robot1.data.joint_names)
        self.motion_body_indexes = self.motion_loader_1.get_body_index(self.robot1.data.body_names)
        self.motion_ref_body_index = self.motion_loader_1.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self.motion_loader_1.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )
        
        # do not lift root height when syncing motions
        if self.cfg.sync_motion:
            self.cfg.init_root_height = 0.0
            self.cfg.episode_length_s = self.motion_loader_1.duration

        # markers
        self.green_markers = VisualizationMarkers(self.cfg.marker_green_cfg)
        self.red_markers = VisualizationMarkers(self.cfg.marker_red_cfg)
        self.green_markers_small = VisualizationMarkers(self.cfg.marker_green_small_cfg)
        self.red_markers_small = VisualizationMarkers(self.cfg.marker_red_small_cfg)

        # set reference motions
        if self.cfg.sync_motion or "imitation" in self.cfg.reward:
            self.ref_state_buffer_length, self.ref_state_buffer_index = self.max_episode_length, 0
            self.ref_state_buffer_1 = {}
            self.ref_state_buffer_2 = {}
            reset_reference_buffer(self, self.motion_loader_1, self.ref_state_buffer_1)
            if self.motion_loader_2: reset_reference_buffer(self, self.motion_loader_2, self.ref_state_buffer_2)
            
        # other properties
        zeros_3dim = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float32)
        zeros_1dim = torch.zeros([self.num_envs, 1], device=self.device, dtype=torch.float32)

        # for center of mass
        self.default_com = None
        self.com_robot1, self.com_robot2 = zeros_3dim.clone(), zeros_3dim.clone()
        self.com_vel_robot1, self.com_vel_robot2 = zeros_3dim.clone(), zeros_3dim.clone()
        self.com_acc_robot1, self.com_acc_robot2 = zeros_3dim.clone(), zeros_3dim.clone()

        # for relative positions
        if self.cfg.require_relative_pose:
            assert self.motion_loader_2 is not None
            self.motion_loader_1.relative_pose = precompute_relative_body_positions(self, source=self.motion_loader_2, target=self.motion_loader_1)
            self.motion_loader_2.relative_pose = precompute_relative_body_positions(self, source=self.motion_loader_1, target=self.motion_loader_2)

        # pairwise joint distance
        if self.cfg.pairwise_joint_distance:
            assert self.motion_loader_2 is not None
            pairwise_joint_distance = compute_pairwise_joint_distance(self, self.motion_loader_1, self.motion_loader_2)
            self.motion_loader_1.pairwise_joint_distance = pairwise_joint_distance

            self.amp_inter_observation_size = self.cfg.num_amp_observations * self.cfg.amp_inter_observation_space
            self.amp_inter_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_inter_observation_size,))
            self.amp_inter_observation_buffer = torch.zeros(
                (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_inter_observation_space), device=self.device
            )
            self.pjd_cfg = {"sqrt": False, "upper_bound": 0.75, "func": "max"}
            
    def _setup_scene(self):
        # add robots
        self.robot1 = Articulation(self.cfg.robot1)
        self.robot2 = Articulation(self.cfg.robot2) if self.cfg.robot2 is not None else None
        self.test_robot = Articulation(self.cfg.test_robot) if self.cfg.test_robot is not None else None

        # add ground plane
        if self.cfg.terrain == "rough":
            TerrainImporter(self.cfg.terrain_cfg)
        else:
            spawn_ground_plane(
                prim_path="/World/ground",
                cfg=GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
            )

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add articulation to scene
        self.scene.articulations["robot1"] = self.robot1
        if self.robot2: self.scene.articulations["robot2"] = self.robot2
        if self.test_robot: self.scene.articulations["test_robot"] = self.test_robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    ### Pre-physics step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.timestep += 1 # (self._sim_step_counter // self.cfg.decimation) is not correct, step is 2 not 1

        if self.cfg.action_clip is not None:
            action_clip = compute_action_clip(self.cfg.action_clip, self.timestep)
            actions = torch.clip(actions, min=-action_clip, max=action_clip) 
        self.actions = actions.clone()

    def _apply_action(self):
        # write reference state to robot1 and 2
        if self.cfg.sync_motion == True:
            assert self.robot2 is not None, "robot2 is None, need 2 robots to sync motions"
            self.write_ref_state(self.robot1, self.ref_state_buffer_1) 
            self.write_ref_state(self.robot2, self.ref_state_buffer_2) 
        # write reference state to test robot
        elif self.cfg.sync_motion == "test_robot":
            assert self.test_robot is not None, f"test_robot is None with sync_motion == 'test_robot'"
            self.write_ref_state(self.test_robot, self.ref_state_buffer_2) 
            target = self.action_offset + self.action_scale * self.actions
            self.robot1.set_joint_position_target(target) # apply action to robot1
        else:
            if self.robot2:
                actions_1, actions_2 = torch.chunk(self.actions, 2, dim=-1)
                target_1 = self.action_offset + self.action_scale * actions_1
                target_2 = self.action_offset + self.action_scale * actions_2
                self.robot1.set_joint_position_target(target_1)
                self.robot2.set_joint_position_target(target_2)
            else:
                target = self.action_offset + self.action_scale * self.actions
                self.robot1.set_joint_position_target(target)
    ### Pre-physics step (End)

    ### Post-physics step
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: # should return resets and time_out
        time_out = self.episode_length_buf >= self.max_episode_length - 1 # bools of envs that are time out
        if self.cfg.early_termination:
            died = self.robot1.data.body_pos_w[:, self.early_termination_body_indexes, 2] < self.termination_heights
            died = torch.max(died, dim=1).values

            # compute falling down angle
            died_1_fall = compute_angle_offset(self, "upward", self.robot1) < 0.3
            died = torch.max(torch.stack([died, died_1_fall], dim=0), dim=0).values
            
            if self.robot2:
                died_2 = self.robot2.data.body_pos_w[:, self.early_termination_body_indexes, 2] < self.termination_heights
                died_2 = torch.max(died_2, dim=1).values
                died_2_fall = compute_angle_offset(self, "upward", self.robot2) < 0.3
                died = torch.max(torch.stack([died, died_2, died_2_fall], dim=0), dim=0).values
            
        else: # no early termination until time out
            died = torch.zeros_like(time_out) 
            
        return died, time_out
    
    def _get_rewards(self) -> torch.Tensor:
        rewards = torch.zeros([self.num_envs], device=self.device)

        if "zero" in self.cfg.reward:
            rewards += reward_zero(self)
        if "stand_forward" in self.cfg.reward:
            rewards += reward_stand_forward(self)
        if "imitation" in self.cfg.reward:
            rewards += reward_imitation(self, loss_function="L2")
        if "min_vel" in self.cfg.reward:
            rewards += reward_min_vel(self)
        if "stand" in self.cfg.reward:
            rewards += reward_stand(self)
        if "com_acc" in self.cfg.reward:
            compute_coms(self)
            rewards += reward_com_acc(self)
        if "energy_penalty" in self.cfg.reward:
            rewards += reward_energy_penalty(self, weight=0.2)
        
        # check NaN in rewards
        nan_envs = check_nan(rewards)
        if torch.any(nan_envs):
            nan_env_ids = torch.nonzero(nan_envs, as_tuple=False).flatten()
            print(f"Warning: NaN detected in rewards {nan_env_ids.tolist()}.")
            rewards[nan_env_ids] = 0.0

        # normalize rewards
        rewards = rewards / len(self.cfg.reward)
        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None): # env_ids: the ids of envs needed to be reset
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot1._ALL_INDICES
        self.robot1.reset(env_ids)
        if self.robot2: self.robot2.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state_1, joint_pos_1, joint_vel_1 = self.reset_strategy_default(env_ids, self.robot1)
            if self.robot2: root_state_2, joint_pos_2, joint_vel_2 = self.reset_strategy_default(env_ids, self.robot2)
        elif self.cfg.reset_strategy.startswith("random"):
            root_state_1, joint_pos_1, joint_vel_1 = self.reset_strategy_random(env_ids, self.motion_loader_1, "start" in self.cfg.reset_strategy)
            if self.robot2: root_state_2, joint_pos_2, joint_vel_2 = self.reset_strategy_random(env_ids, self.motion_loader_2, "start" in self.cfg.reset_strategy)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")
        
        # reset robot 1
        self.robot1.write_root_link_pose_to_sim(root_state_1[:, :7], env_ids)
        self.robot1.write_root_com_velocity_to_sim(root_state_1[:, 7:], env_ids)
        self.robot1.write_joint_state_to_sim(joint_pos_1, joint_vel_1, None, env_ids)
        # reset robot 2
        if self.robot2:
            self.robot2.write_root_link_pose_to_sim(root_state_2[:, :7], env_ids)
            self.robot2.write_root_com_velocity_to_sim(root_state_2[:, 7:], env_ids)
            self.robot2.write_joint_state_to_sim(joint_pos_2, joint_vel_2, None, env_ids)

        # reset center of mass
        if "com" in self.cfg.reward:
            self.com_robot1 = compute_whole_body_com(self, self.robot1)[env_ids]
            self.com_vel_robot1[env_ids] = 0.0
            self.com_acc_robot1[env_ids] = 0.0
            
    def _get_observations(self) -> dict:
        # build task observation
        obs_1 = compute_obs(self,
            self.robot1.data.joint_pos,
            self.robot1.data.joint_vel,
            self.robot1.data.body_pos_w[:, self.ref_body_index],
            self.robot1.data.body_quat_w[:, self.ref_body_index],
            self.robot1.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot1.data.body_ang_vel_w[:, self.ref_body_index],
        )
        if self.robot2:
            obs_2 = compute_obs(self,
                self.robot2.data.joint_pos,
                self.robot2.data.joint_vel,
                self.robot2.data.body_pos_w[:, self.ref_body_index],
                self.robot2.data.body_quat_w[:, self.ref_body_index],
                self.robot2.data.body_lin_vel_w[:, self.ref_body_index],
                self.robot2.data.body_ang_vel_w[:, self.ref_body_index],
            )

        # detect NaN in observations
        nan_envs = check_nan(obs_1)
        if self.robot2: 
            nan_envs_2 = check_nan(obs_2)
            nan_envs = torch.logical_or(nan_envs, nan_envs_2)
        
        # reset NaN environments
        if torch.any(nan_envs):
            nan_env_ids = torch.nonzero(nan_envs, as_tuple=False).flatten()
            if nan_env_ids.shape[0] == self.num_envs:
                print("All environments are NaN, training process ends.")
                sys.exit(0)
            print(f"Warning: NaN detected in envs {nan_env_ids.tolist()}, resetting these envs.")
            self._reset_idx(nan_env_ids)
            
            if len(nan_env_ids) > 0:
                obs_1[nan_env_ids] = compute_obs(self,
                    self.robot1.data.joint_pos[nan_env_ids],
                    self.robot1.data.joint_vel[nan_env_ids],
                    self.robot1.data.body_pos_w[nan_env_ids, self.ref_body_index],
                    self.robot1.data.body_quat_w[nan_env_ids, self.ref_body_index],
                    self.robot1.data.body_lin_vel_w[nan_env_ids, self.ref_body_index],
                    self.robot1.data.body_ang_vel_w[nan_env_ids, self.ref_body_index],
                )
                if self.robot2: 
                    obs_2[nan_env_ids] = compute_obs(self,
                        self.robot2.data.joint_pos[nan_env_ids],
                        self.robot2.data.joint_vel[nan_env_ids],
                        self.robot2.data.body_pos_w[nan_env_ids, self.ref_body_index],
                        self.robot2.data.body_quat_w[nan_env_ids, self.ref_body_index],
                        self.robot2.data.body_lin_vel_w[nan_env_ids, self.ref_body_index],
                        self.robot2.data.body_ang_vel_w[nan_env_ids, self.ref_body_index],
                    )
        
        # input states with pose of another character
        if self.cfg.require_another_pose:
            obs_1, obs_2 = torch.cat([obs_1, obs_2], dim=-1), torch.cat([obs_2, obs_1], dim=-1)

        # if amp obs is different from obs
        amp_obs_1 = compute_obs(self,
            self.robot1.data.joint_pos,
            self.robot1.data.joint_vel,
            self.robot1.data.body_pos_w[:, self.ref_body_index],
            self.robot1.data.body_quat_w[:, self.ref_body_index],
            self.robot1.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot1.data.body_ang_vel_w[:, self.ref_body_index],
        )
        if self.robot2:
            amp_obs_2 = compute_obs(self,
                self.robot2.data.joint_pos,
                self.robot2.data.joint_vel,
                self.robot2.data.body_pos_w[:, self.ref_body_index],
                self.robot2.data.body_quat_w[:, self.ref_body_index],
                self.robot2.data.body_lin_vel_w[:, self.ref_body_index],
                self.robot2.data.body_ang_vel_w[:, self.ref_body_index],
            )
        
        # combine 2 robots observation 
        if self.robot2: 
            obs = torch.cat([obs_1, obs_2], dim=-1)
            amp_obs = torch.cat([amp_obs_1, amp_obs_2], dim=-1)
        else: 
            obs = obs_1
            amp_obs = amp_obs_1

        # update AMP observation history (pop out)
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # update AMP observation history (push in)
        self.amp_observation_buffer[:, 0] = amp_obs.clone() # buffer: [num_envs, num_amp_observations, amp_observation_space]
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        # interaction observation
        if self.cfg.pairwise_joint_distance:
            assert self.test_robot is not None, "test_robot is not None"
            pairwise_joint_distance = compute_pairwise_joint_distance(self, self.robot1, self.test_robot)
            obs = torch.cat([obs, pairwise_joint_distance], dim=-1)
            amp_inter_obs = pairwise_joint_distance.clone()

            # update interaction observation history (pop out)
            for i in reversed(range(self.cfg.num_amp_observations - 1)):
                self.amp_inter_observation_buffer[:, i + 1] = self.amp_inter_observation_buffer[:, i]
            # update interaction observation history (push in)
            self.amp_inter_observation_buffer[:, 0] = amp_inter_obs.clone() # buffer: [num_envs, num_amp_inter_observations, amp_inter_observation_space]
            self.extras["amp_interaction_obs"] = self.amp_inter_observation_buffer.view(-1, self.amp_inter_observation_size)

            # get interaction reward weights 
            pairwise_joint_distance = self.motion_loader_1.get_pairwise_joint_distance(frame=self.episode_length_buf)
            pairwise_joint_distance = torch.norm(pairwise_joint_distance[:, 23*3:].reshape(self.num_envs, len(self.key_body_indexes)**2, 3), dim=-1) # test: extract relative positions
            interaction_reward_weights = compute_pairwise_joint_distance_env_weight(pairwise_joint_distance, func=self.pjd_cfg["func"], sqrt=self.pjd_cfg["sqrt"], upper_bound=self.pjd_cfg["upper_bound"]).view(self.num_envs, -1)
            self.extras["interaction_reward_weights"] = interaction_reward_weights

            # animate_pairwise_joint_distance_heatmap(pairwise_joint_distance[0], key_names=self.cfg.key_body_names, sqrt=self.pjd_cfg["sqrt"], upper_bound=self.pjd_cfg["upper_bound"])


        return {"policy": obs}
    ### Post-physics step (End)
        
    # reset strategies
    def reset_strategy_default(self, env_ids: torch.Tensor, robot: Articulation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] += self.cfg.init_root_height  # lift the humanoid slightly to avoid collisions with the ground
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_vel = robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def reset_strategy_random(
        self, env_ids: torch.Tensor, motion_loader: MotionLoader, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # env_ids: the ids of envs to be reset
        num_samples = env_ids.shape[0]

        # sample random motion times (or zeros if start is True)
        if motion_loader == self.motion_loader_1: # only sample once for both robots
            self.sample_times = np.zeros(num_samples) if start else motion_loader.sample_times(num_samples, upper_bound=0.95)
        
        # for imitation reward, use self.episode_length_buf as frame index in dataset
        if "imitation" in self.cfg.reward:
            self.episode_length_buf[env_ids] = torch.from_numpy(motion_loader._get_frame_index_from_time(self.sample_times)[0]).long().to(self.device)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            root_linear_velocity,
            root_angular_velocity,
        ) = motion_loader.sample(num_samples=num_samples, times=self.sample_times)
        
        # get root transforms (the humanoid torso)
        motion_root_index = motion_loader.get_body_index([self.cfg.reference_body])[0]
        root_state = self.robot1.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_root_index] + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = body_rotations[:, motion_root_index]
        root_state[:, 7:10] = root_linear_velocity
        root_state[:, 10:13] = root_angular_velocity
        root_state[:, 2] += self.cfg.init_root_height  # lift the humanoid slightly to avoid collisions with the ground
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation buffer after resetting environments
        amp_observations = self.collect_reference_motions(num_samples, self.sample_times, self.motion_loader_1)
        if self.robot2: 
            amp_observations_2 = self.collect_reference_motions(num_samples, self.sample_times, self.motion_loader_2)
            amp_observations = torch.cat([amp_observations, amp_observations_2], dim=-1)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        # update interaction observation buffer after resetting environments
        if self.cfg.pairwise_joint_distance: 
            amp_inter_observations = self.collect_reference_interactions(num_samples, self.sample_times)
            self.amp_inter_observation_buffer[env_ids] = amp_inter_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None, motion_loader=None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self.motion_loader_1.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self.motion_loader_1.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()

        # update AMP observation buffer after resetting environments
        if motion_loader: 
            # get motions
            (
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                root_linear_velocity,
                root_angular_velocity,
            ) = motion_loader.sample(num_samples=num_samples, times=times)
            
            # compute AMP observation
            amp_observation = compute_obs(self,
                dof_positions[:, self.motion_dof_indexes],
                dof_velocities[:, self.motion_dof_indexes],
                body_positions[:, self.motion_ref_body_index],
                body_rotations[:, self.motion_ref_body_index],
                root_linear_velocity,
                root_angular_velocity,
            ).view(-1, int(self.amp_observation_size/2) if self.robot2 else self.amp_observation_size)

            return amp_observation # (num_envs, 2 * obs)

        # update AMP motion dataset (ground truth) for agent
        else: 
            motion_loader = self.motion_loader_1
            # get motions
            (
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                root_linear_velocity,
                root_angular_velocity,
            ) = motion_loader.sample(num_samples=num_samples, times=times)

            # compute AMP observation
            amp_observation = compute_obs(self,
                dof_positions[:, self.motion_dof_indexes],
                dof_velocities[:, self.motion_dof_indexes],
                body_positions[:, self.motion_ref_body_index],
                body_rotations[:, self.motion_ref_body_index],
                root_linear_velocity,
                root_angular_velocity,
            ).view(-1, int(self.amp_observation_size/2) if self.robot2 else self.amp_observation_size)
            
            if self.robot2:
                motion_loader = self.motion_loader_2
                # get motions
                (
                    dof_positions,
                    dof_velocities,
                    body_positions,
                    body_rotations,
                    root_linear_velocity,
                    root_angular_velocity,
                ) = motion_loader.sample(num_samples=num_samples, times=times)
                # compute AMP observation
                amp_observation_2 = compute_obs(self,
                    dof_positions[:, self.motion_dof_indexes],
                    dof_velocities[:, self.motion_dof_indexes],
                    body_positions[:, self.motion_ref_body_index],
                    body_rotations[:, self.motion_ref_body_index],
                    root_linear_velocity,
                    root_angular_velocity,
                ).view(-1, int(self.amp_observation_size/2) if self.robot2 else self.amp_observation_size)
                amp_observation = torch.cat([amp_observation, amp_observation_2], dim=-1)

            return amp_observation
        
    def collect_reference_interactions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self.motion_loader_1.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self.motion_loader_1.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()

        # update interaction observation buffer after resetting environments
        #TODO: move joint selection here
        pairwise_joint_distance = self.motion_loader_1.get_pairwise_joint_distance(times=current_times)
        # pairwise_joint_distance = compute_pairwise_joint_distance_weight(pairwise_joint_distance) # test: temporarily disable
        amp_inter_observation = pairwise_joint_distance.reshape(-1, self.amp_inter_observation_size) # [envs, interaction_space]

        return amp_inter_observation # (num_envs, 2 * obs)
    
    def write_ref_state(self, robot: Articulation, ref_state_buffer):
        root_pos_quat = ref_state_buffer['root_state'][self.episode_length_buf, :7]
        root_pos_quat[:, :3] += self.scene.env_origins

        robot.write_root_link_pose_to_sim(root_pos_quat,
                                          robot._ALL_INDICES)
        robot.write_root_com_velocity_to_sim(ref_state_buffer['root_state'][self.episode_length_buf, 7:],
                                             robot._ALL_INDICES)
        
        # what is the difference between the two lines below?
        # self.robot.write_root_pose_to_sim(self.ref_state_buffer['root_state'][:, self.ref_state_buffer_index, :7], 
        #                                        self.robot._ALL_INDICES)
        # self.robot.write_root_state_to_sim(self.ref_state_buffer['root_state'][:, self.ref_state_buffer_index], 
        #                                           self.robot._ALL_INDICES)
        
        robot.write_joint_state_to_sim(ref_state_buffer["dof_pos"][self.episode_length_buf],
                                       ref_state_buffer["dof_vel"][self.episode_length_buf],
                                       None, robot._ALL_INDICES)


