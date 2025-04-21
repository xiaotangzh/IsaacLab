# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate
from .task_cfg import Cfg
from .motions.motion_loader_smpl import MotionLoader as MotionLoaderSMPL
from .motions.motion_loader_humanoid import MotionLoader as MotionLoaderHumanoid
import sys
import random

# marker
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
import isaaclab.sim as sim_utils

# terrain
from isaaclab.terrains import TerrainImporter

# compute relative position
import isaaclab.utils.math as math_utils

class Env(DirectRLEnv):
    cfg: Cfg

    def __init__(self, cfg: Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot1.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot1.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        self.action_clip = self.cfg.action_clip
        self.termination_heights = torch.tensor(self.cfg.termination_heights, device=self.device)

        # load motion
        if self.cfg.robot_format == "humanoid": MotionLoader = MotionLoaderHumanoid
        else: MotionLoader = MotionLoaderSMPL
        self._motion_loader_1 = MotionLoader(motion_file=self.cfg.motion_file_1, device=self.device)
        self._motion_loader_2 = MotionLoader(motion_file=self.cfg.motion_file_2, device=self.device) if hasattr(self.cfg, "motion_file_2") else None 
        self.sample_times = None # synchronize sampling times for two robots
        if self.cfg.episode_length_s < 0 or self.cfg.episode_length_s > self._motion_loader_1.duration:
            self.cfg.episode_length_s = self._motion_loader_1.duration

        # DOF and key body indexes
        key_body_names = self.cfg.key_body_names
        self.ref_body_index = self.robot1.data.body_names.index(self.cfg.reference_body)
        self.early_termination_body_indexes = [self.robot1.data.body_names.index(name) for name in self.cfg.termination_bodies]
        self.key_body_indexes = [self.robot1.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader_1.get_dof_index(self.robot1.data.joint_names)
        self.motion_ref_body_index = self._motion_loader_1.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader_1.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )
        
        # do not lift root height in some tasks
        if "imitation" in self.cfg.reward or self.cfg.sync_motion:
            self.cfg.init_root_height = 0.0
            self.cfg.episode_length_s = self._motion_loader_1.duration
        
        # sync motion
        if self.cfg.sync_motion:
            self.ref_state_buffer_length, self.ref_state_buffer_index = self.max_episode_length, 0
            self.ref_state_buffer_1 = {}
            self.ref_state_buffer_2 = {}
            self.reset_reference_buffer(self._motion_loader_1, self.ref_state_buffer_1)
            if hasattr(self.cfg, "robot2"): self.reset_reference_buffer(self._motion_loader_2, self.ref_state_buffer_2)

        # other properties
        zeros_3dim = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float32)
        zeros_1dim = torch.zeros([self.num_envs, 1], device=self.device, dtype=torch.float32)

        # for center of mass
        self.default_com = None
        self.com_robot1, self.com_robot2 = zeros_3dim.clone(), zeros_3dim.clone()
        self.com_vel_robot1 = zeros_3dim.clone()
        self.com_acc_robot1 = zeros_3dim.clone()

        # for standing reward
        self.current_root_forward_offset = zeros_1dim.clone()
        self.current_root_upward_offset = zeros_1dim.clone() #TODO: robot 2

        # markers
        # self.green_markers = VisualizationMarkers(self.cfg.marker_green_cfg)
        self.red_markers = VisualizationMarkers(self.cfg.marker_red_cfg)

        # for relative positions
        if self.cfg.require_relative_pose:
            assert self._motion_loader_2 is not None
            self._motion_loader_1.relative_pose = self.precompute_relative_body_positions(source=self._motion_loader_2, target=self._motion_loader_1)
            self._motion_loader_2.relative_pose = self.precompute_relative_body_positions(source=self._motion_loader_1, target=self._motion_loader_2)
            
    def _setup_scene(self):
        self.robot1 = Articulation(self.cfg.robot1)
        self.robot2 = Articulation(self.cfg.robot2) if hasattr(self.cfg, "robot2") else None
        # add ground plane
        if self.cfg.terrain == "uneven":
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
        if self.robot2:
            self.scene.articulations["robot2"] = self.robot2
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    ### Pre-physics step
    def _pre_physics_step(self, actions: torch.Tensor):
        if self.action_clip[0] and self.action_clip[1]:
            actions = torch.clip(actions, min=self.action_clip[0], max=self.action_clip[1]) # clip the actions
        self.actions = actions.clone()

        # visualize markers
        # if self.default_com is not None: 
        #     print(self.default_com.shape)
        #     self.green_markers.visualize(translations=self.default_com)
        if self.robot2:
            self.red_markers.visualize(translations=torch.cat([self.com_robot1, self.com_robot2], dim=0))
        else:
            self.red_markers.visualize(translations=self.com_robot1)

    def _apply_action(self):
        if self.cfg.sync_motion:
            self.write_ref_state(self.robot1, self.ref_state_buffer_1) 
            if self.robot2: self.write_ref_state(self.robot2, self.ref_state_buffer_2)
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
            died_1 = self.robot1.data.body_pos_w[:, self.early_termination_body_indexes, 2] < self.termination_heights
            died_1 = torch.max(died_1, dim=1).values

            # compute falling down angle
            died_1_fall = self.compute_angle_offset("upward", self.robot1) < 0.3 #TODO: robot2 
            died_1 = torch.max(torch.stack([died_1, died_1_fall], dim=0), dim=0).values
            
            if self.robot2:
                died_2 = self.robot2.data.body_pos_w[:, self.early_termination_body_indexes, 2] < self.termination_heights
                died_2 = torch.max(died_2, dim=1).values
                died = torch.max(torch.stack([died_1, died_2], dim=0), dim=0).values
            else:
                died = died_1
            
        else: # no early termination until time out
            died = torch.zeros_like(time_out) 
        
        # end of reference buffer
        if self.cfg.sync_motion and self.ref_state_buffer_index >= self.ref_state_buffer_length: #TODO: could be refined
            died = torch.ones_like(time_out)
            
        # if reward is imitation, reset envs that are longer than the dataset frame length
        if self.cfg.reward == "imitation":
            died_imitation = self.episode_length_buf >= self._motion_loader_1.num_frames
            died = torch.max(torch.stack([died, died_imitation], dim=0), dim=0).values
            
        return died, time_out
    
    #TODO: adapt to two robots
    def _get_rewards(self) -> torch.Tensor:
        rewards = torch.zeros([self.num_envs], device=self.device)

        if "ones" in self.cfg.reward:
            rewards += self.reward_ones()
        if "stand_forward" in self.cfg.reward:
            rewards += self.reward_stand_forward()
        if "imitation" in self.cfg.reward:
            rewards += self.reward_imitation()
        if "min_vel" in self.cfg.reward:
            rewards += self.reward_min_vel()
        if "stand" in self.cfg.reward:
            rewards += self.reward_stand()
        if "com_acc" in self.cfg.reward:
            self.compute_coms()
            rewards += self.reward_com_acc()
        
        nan_envs = check_nan(rewards)
        if torch.any(nan_envs):
            nan_env_ids = torch.nonzero(nan_envs, as_tuple=False).flatten()
            print(f"Warning: NaN detected in rewards {nan_env_ids.tolist()}.")
            rewards[nan_env_ids] = 0.0
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
            start = "start" in self.cfg.reset_strategy
            root_state_1, joint_pos_1, joint_vel_1 = self.reset_strategy_random(env_ids, self._motion_loader_1, start)
            if self.robot2: root_state_2, joint_pos_2, joint_vel_2 = self.reset_strategy_random(env_ids, self._motion_loader_2, start)

            # update AMP observation buffer after resetting environments
            num_samples = env_ids.shape[0]
            amp_observations = self.collect_reference_motions(num_samples, self.sample_times, self._motion_loader_1)
            if self.robot2: 
                amp_observations_2 = self.collect_reference_motions(num_samples, self.sample_times, self._motion_loader_2)
                amp_observations = torch.cat([amp_observations, amp_observations_2], dim=-1)
            self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")
        
        # reset robot 1
        self.robot1.write_root_link_pose_to_sim(root_state_1[:, :7], env_ids)
        self.robot1.write_root_com_velocity_to_sim(root_state_1[:, 7:], env_ids)
        self.robot1.write_joint_state_to_sim(joint_pos_1, joint_vel_1, None, env_ids)
        if self.robot2:
            # reset robot 2
            self.robot2.write_root_link_pose_to_sim(root_state_2[:, :7], env_ids)
            self.robot2.write_root_com_velocity_to_sim(root_state_2[:, 7:], env_ids)
            self.robot2.write_joint_state_to_sim(joint_pos_2, joint_vel_2, None, env_ids)

        # reset center of mass
        if "com" in self.cfg.reward:
            self.com_robot1 = self.compute_whole_body_com(self.robot1)[env_ids]
            self.com_vel_robot1[env_ids] = 0.0
            self.com_acc_robot1[env_ids] = 0.0
            
    def _get_observations(self) -> dict:
        # build task observation
        obs_1 = self.compute_obs(
            self.robot1.data.joint_pos,
            self.robot1.data.joint_vel,
            self.robot1.data.body_pos_w[:, self.ref_body_index],
            self.robot1.data.body_quat_w[:, self.ref_body_index],
            self.robot1.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot1.data.body_ang_vel_w[:, self.ref_body_index],
            self._motion_loader_1.get_relative_pose(frame=self.episode_length_buf) if self.cfg.require_relative_pose else None
        )
        if self.robot2:
            obs_2 = self.compute_obs(
                self.robot2.data.joint_pos,
                self.robot2.data.joint_vel,
                self.robot2.data.body_pos_w[:, self.ref_body_index],
                self.robot2.data.body_quat_w[:, self.ref_body_index],
                self.robot2.data.body_lin_vel_w[:, self.ref_body_index],
                self.robot2.data.body_ang_vel_w[:, self.ref_body_index],
                self._motion_loader_2.get_relative_pose(frame=self.episode_length_buf) if self.cfg.require_relative_pose else None
            )

        # detect NaN in observations
        nan_envs = check_nan(obs_1)
        if self.robot2: 
            nan_envs_2 = check_nan(obs_2)
            nan_envs = torch.logical_or(nan_envs, nan_envs_2)
        
        # reset NaN environments
        if torch.any(nan_envs):
            nan_env_ids = torch.nonzero(nan_envs, as_tuple=False).flatten()
            print(f"Warning: NaN detected in envs {nan_env_ids.tolist()}, resetting these envs.")
            self._reset_idx(nan_env_ids)
            
            if len(nan_env_ids) > 0:
                obs_1[nan_env_ids] = self.compute_obs(
                    self.robot1.data.joint_pos[nan_env_ids],
                    self.robot1.data.joint_vel[nan_env_ids],
                    self.robot1.data.body_pos_w[nan_env_ids, self.ref_body_index],
                    self.robot1.data.body_quat_w[nan_env_ids, self.ref_body_index],
                    self.robot1.data.body_lin_vel_w[nan_env_ids, self.ref_body_index],
                    self.robot1.data.body_ang_vel_w[nan_env_ids, self.ref_body_index],
                    self._motion_loader_1.get_relative_pose(frame=self.episode_length_buf[nan_env_ids]) if self.cfg.require_relative_pose else None
                )
                if self.robot2: 
                    obs_2[nan_env_ids] = self.compute_obs(
                        self.robot2.data.joint_pos[nan_env_ids],
                        self.robot2.data.joint_vel[nan_env_ids],
                        self.robot2.data.body_pos_w[nan_env_ids, self.ref_body_index],
                        self.robot2.data.body_quat_w[nan_env_ids, self.ref_body_index],
                        self.robot2.data.body_lin_vel_w[nan_env_ids, self.ref_body_index],
                        self.robot2.data.body_ang_vel_w[nan_env_ids, self.ref_body_index],
                        self._motion_loader_2.get_relative_pose(frame=self.episode_length_buf[nan_env_ids]) if self.cfg.require_relative_pose else None
                    )
        
        # if input states with pose of another character
        if self.cfg.require_another_pose:
            obs_1, obs_2 = torch.cat([obs_1, obs_2], dim=-1), torch.cat([obs_2, obs_1], dim=-1)

        # if amp obs is different from obs
        amp_obs_1 = self.compute_obs(
            self.robot1.data.joint_pos,
            self.robot1.data.joint_vel,
            self.robot1.data.body_pos_w[:, self.ref_body_index],
            self.robot1.data.body_quat_w[:, self.ref_body_index],
            self.robot1.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot1.data.body_ang_vel_w[:, self.ref_body_index],
            self.compute_relative_body_positions(source=self.robot2, target=self.robot1) if self.cfg.require_relative_pose else None
        )
        if self.robot2:
            amp_obs_2 = self.compute_obs(
                self.robot2.data.joint_pos,
                self.robot2.data.joint_vel,
                self.robot2.data.body_pos_w[:, self.ref_body_index],
                self.robot2.data.body_quat_w[:, self.ref_body_index],
                self.robot2.data.body_lin_vel_w[:, self.ref_body_index],
                self.robot2.data.body_ang_vel_w[:, self.ref_body_index],
                self.compute_relative_body_positions(source=self.robot1, target=self.robot2) if self.cfg.require_relative_pose else None
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
        self, env_ids: torch.Tensor, motion_loader, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # env_ids: the ids of envs to be reset
        num_samples = env_ids.shape[0]

        # sample random motion times (or zeros if start is True)
        if motion_loader == self._motion_loader_1: # only sample once for both robots
            self.sample_times = np.zeros(num_samples) if start else motion_loader.sample_times(num_samples)
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

        return root_state, dof_pos, dof_vel

    # Collect ground truth observations, used in agent
    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None, motion_loader=None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader_1.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader_1.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()

        if motion_loader: # updating AMP observation buffer
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
            amp_observation = self.compute_obs(
                dof_positions[:, self.motion_dof_indexes],
                dof_velocities[:, self.motion_dof_indexes],
                body_positions[:, self.motion_ref_body_index],
                body_rotations[:, self.motion_ref_body_index],
                root_linear_velocity,
                root_angular_velocity,
                motion_loader.get_relative_pose(times=current_times) if self.cfg.require_relative_pose else None
            ).view(-1, int(self.amp_observation_size/2) if self.robot2 else self.amp_observation_size)

            return amp_observation # (num_envs, state transitions)

        else: # updating AMP motion dataset (ground truth) for agent
            motion_loader = self._motion_loader_1
            # get motions
            (
                dof_positions,
                dof_velocities,
                body_positions,
                body_rotations,
                root_linear_velocity,
                root_angular_velocity,
            ) = motion_loader.sample(num_samples=num_samples, times=times) # (num_samples * num_amp_observation, dof)

            # compute AMP observation
            amp_observation = self.compute_obs(
                dof_positions[:, self.motion_dof_indexes],
                dof_velocities[:, self.motion_dof_indexes],
                body_positions[:, self.motion_ref_body_index],
                body_rotations[:, self.motion_ref_body_index],
                root_linear_velocity,
                root_angular_velocity,
                motion_loader.get_relative_pose(times=current_times) if self.cfg.require_relative_pose else None
            ).view(-1, int(self.amp_observation_size/2) if self.robot2 else self.amp_observation_size)
            
            if self.robot2:
                motion_loader = self._motion_loader_2
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
                amp_observation_2 = self.compute_obs(
                    dof_positions[:, self.motion_dof_indexes],
                    dof_velocities[:, self.motion_dof_indexes],
                    body_positions[:, self.motion_ref_body_index],
                    body_rotations[:, self.motion_ref_body_index],
                    root_linear_velocity,
                    root_angular_velocity,
                    motion_loader.get_relative_pose(times=current_times) if self.cfg.require_relative_pose else None
                ).view(-1, int(self.amp_observation_size/2) if self.robot2 else self.amp_observation_size)
                amp_observation = torch.cat([amp_observation, amp_observation_2], dim=-1)

            return amp_observation
    
    def reset_reference_buffer(self, motion_loader, ref_state_buffer: dict, env_ids: torch.Tensor | None=None):
        env_ids = self.robot1._ALL_INDICES if env_ids is None else env_ids
        num_samples = env_ids.shape[0]
        
        # sample reference actions
        (
            ref_dof_positions,
            ref_dof_velocities,
            ref_body_positions,
            ref_body_rotations,
            ref_root_linear_velocity,
            ref_root_angular_velocity,
        ) = motion_loader.get_all_references(num_samples)
        
        ref_root_state = self.robot1.data.default_root_state[env_ids].unsqueeze(1).expand(-1, ref_dof_positions.shape[1], -1).clone()
        ref_root_state[:, :, 0:3] = ref_body_positions[:, :, self.motion_ref_body_index] + self.scene.env_origins[env_ids].unsqueeze(1)
        ref_root_state[:, :, 3:7] = ref_body_rotations[:, :, self.motion_ref_body_index]
        ref_root_state[:, :, 7:10] = ref_root_linear_velocity
        ref_root_state[:, :, 10:13] = ref_root_angular_velocity
        
        # set reference buffer
        ref_state_buffer.update({
            "root_state": ref_root_state,
            "joint_pos": ref_dof_positions[:, :, self.motion_dof_indexes],
            "joint_vel": ref_dof_velocities[:, :, self.motion_dof_indexes],
        })
    
    def write_ref_state(self, robot, ref_state_buffer):
        robot.write_root_link_pose_to_sim(ref_state_buffer['root_state'][torch.arange(self.num_envs), self.episode_length_buf, :7],
                                          robot._ALL_INDICES)
        robot.write_root_com_velocity_to_sim(ref_state_buffer['root_state'][torch.arange(self.num_envs), self.episode_length_buf, 7:],
                                             robot._ALL_INDICES)
        
        #TODO: what is the difference between the two lines below?
        # self.robot.write_root_pose_to_sim(self.ref_state_buffer['root_state'][:, self.ref_state_buffer_index, :7], 
        #                                        self.robot._ALL_INDICES)
        # self.robot.write_root_state_to_sim(self.ref_state_buffer['root_state'][:, self.ref_state_buffer_index], 
        #                                           self.robot._ALL_INDICES)
        
        robot.write_joint_state_to_sim(ref_state_buffer['joint_pos'][torch.arange(self.num_envs), self.episode_length_buf],
                                       ref_state_buffer['joint_vel'][torch.arange(self.num_envs), self.episode_length_buf],
                                       None, robot._ALL_INDICES)
        
    def precompute_relative_body_positions(self, 
                                           source: MotionLoaderSMPL | MotionLoaderHumanoid,
                                           target: MotionLoaderSMPL | MotionLoaderHumanoid) -> torch.Tensor:
        source_body_positions = source.get_all_references()[2][0] # (frames, body num, 3)
        target_root_positions = target.get_all_references()[2][0][:,self.motion_ref_body_index] # (frames, 3)
        target_root_rotations = target.get_all_references()[3][0][:,self.motion_ref_body_index] # (frames, 4)
        return math_utils.transform_points(points=source_body_positions, pos=target_root_positions, quat=target_root_rotations)
    
    def compute_relative_body_positions(self, 
                                        source: Articulation,
                                        target: Articulation) -> torch.Tensor:
        source_body_positions = source.data.body_pos_w # (envs, body num, 3)
        target_root_positions = target.data.body_pos_w[:,self.ref_body_index] # (envs, 3)
        target_root_rotations = target.data.body_quat_w[:,self.ref_body_index] # (envs, 4)
        return math_utils.transform_points(points=source_body_positions, pos=target_root_positions, quat=target_root_rotations)

    def reward_stand_forward(self) -> torch.Tensor:
        angle_offset = self.compute_angle_offset("forward", self.robot1) 
        reward = torch.abs(angle_offset)
        if self.robot2: 
            angle_offset2 = self.compute_angle_offset("forward", self.robot2)
            reward = (reward + torch.abs(angle_offset2)) / 2
        # print(f"stand forward reward: {torch.mean(reward)}")
        return reward
    
    def reward_min_vel(self) -> torch.Tensor: # [0, 3]
        reward = torch.mean(self.robot1.data.joint_vel, dim=-1)
        if self.robot2: 
            reward = (reward + torch.mean(self.robot2.data.joint_vel, dim=-1)) / 2
        reward = torch.clip((reward * (-1) + 3) / 3, min=0, max=1)
        # print(f"min vel reward: {torch.mean(reward)}")
        return reward
    
    def reward_stand(self) -> torch.Tensor:
        reward = torch.clip(self.robot1.data.body_pos_w[:, self.ref_body_index, 2], min=0, max=self.robot1.data.default_root_state[0, 2].item())
        if self.robot2:
            reward = (reward + torch.clip(self.robot2.data.body_pos_w[:, self.ref_body_index, 2], min=0, max=self.robot2.data.default_root_state[0, 2].item())) / 2
        # print(f"stand reward: {torch.mean(reward)}")
        return reward
    
    def reward_com(self) -> torch.Tensor:
        self.com_robot1 = self.compute_whole_body_com(self.robot1)
        reward = 1 - torch.mean(torch.abs(self.com_robot1 - self.default_com))
        # TODO: robot2
        # print(f"center of mass reward: {torch.mean(reward)}")
        return reward
    
    def reward_com_acc(self, decay: float=0.01) -> torch.Tensor:
        reward = torch.clip(torch.exp(-decay * torch.mean(torch.abs(self.com_acc_robot1))), min=0.0, max=1.0)
        # print(f"center of mass acc reward: {reward}")
        return reward
    
    # def compute_coms(self):
    #     current_com_vel_robot1, current_com_acc_robot1 = None, None
    #     current_com_robot1 = self.compute_whole_body_com(self.robot1)
    #     if self.com_robot1 is not None: 
    #         current_com_vel_robot1 = (current_com_robot1 - self.com_robot1) / self.cfg.dt
    #         if self.com_vel_robot1 is not None:
    #             current_com_acc_robot1 = (current_com_vel_robot1 - self.com_vel_robot1) / self.cfg.dt
    #     # update coms    
    #     self.com_robot1, self.com_vel_robot1, self.com_acc_robot1 = current_com_robot1, current_com_vel_robot1, current_com_acc_robot1

    #     if self.robot2:
    #         current_com_vel_robot2, current_com_acc_robot2 = None, None
    #         current_com_robot2 = self.compute_whole_body_com(self.robot2)
    #         if self.com_robot2 is not None: 
    #             current_com_vel_robot2 = (current_com_robot2 - self.com_robot2) / self.cfg.dt
    #             if self.com_vel_robot2 is not None:
    #                 current_com_acc_robot2 = (current_com_vel_robot2 - self.com_vel_robot2) / self.cfg.dt
    #         # update coms    
    #         self.com_robot2, self.com_vel_robot2, self.com_acc_robot2 = current_com_robot2, current_com_vel_robot2, current_com_acc_robot2

    def compute_coms(self):
        current_com_robot1 = self.compute_whole_body_com(self.robot1)
        current_com_vel_robot1 = (current_com_robot1 - self.com_robot1) / self.cfg.dt
        current_com_acc_robot1 = (current_com_vel_robot1 - self.com_vel_robot1) / self.cfg.dt
        # update coms    
        self.com_robot1, self.com_vel_robot1, self.com_acc_robot1 = current_com_robot1, current_com_vel_robot1, current_com_acc_robot1

    def compute_whole_body_com(self, robot: Articulation) -> torch.Tensor:
        body_masses = robot.data.default_mass.to(self.device)
        
        body_com_positions = robot.data.body_com_pos_w
        
        total_mass = body_masses.sum(dim=1, keepdim=True)  # [num_envs, 1]
        weighted_positions = body_com_positions * body_masses.unsqueeze(-1)  # [num_envs, num_bodies, 3]
        whole_body_com = weighted_positions.sum(dim=1) / total_mass  # [num_envs, 3]
        
        return whole_body_com
    
    def compute_angle_offset(self, target_direction, robot: Articulation) -> torch.Tensor:
        if target_direction == "forward":
            target_direction = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            idx = 0
        elif target_direction == "leftward":
            target_direction = torch.tensor([0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            idx = 1
        elif target_direction == "upward":
            target_direction = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            idx = 2

        current_quat = robot.data.body_quat_w[:, self.ref_body_index]
        current_quat = current_quat / torch.norm(current_quat, dim=-1, keepdim=True)
        current_direction = quat_rotate(current_quat, target_direction)
        angle_offset = current_direction[:, idx]  # [-1, 1] from opposite to same direction as target
        return angle_offset
    
    def reward_imitation(self) -> torch.Tensor:
        frames = self._motion_loader_1.num_frames
        
        obs1 = torch.cat([self.robot1.data.body_pos_w[:, self.ref_body_index],
                        self.robot1.data.body_quat_w[:, self.ref_body_index],
                        self.robot1.data.body_lin_vel_w[:, self.ref_body_index],
                        self.robot1.data.body_ang_vel_w[:, self.ref_body_index],
                        self.robot1.data.joint_pos,
                        self.robot1.data.joint_vel], dim=-1)

        ref1 = torch.cat([self.ref_state_buffer_1["root_state"],
                          self.ref_state_buffer_1["joint_pos"].reshape(self.num_envs, frames, -1),
                          self.ref_state_buffer_1["joint_vel"].reshape(self.num_envs, frames, -1)], dim=-1)
        
        obs2 = torch.cat([self.robot2.data.body_pos_w[:, self.ref_body_index],
                        self.robot2.data.body_quat_w[:, self.ref_body_index],
                        self.robot2.data.body_lin_vel_w[:, self.ref_body_index],
                        self.robot2.data.body_ang_vel_w[:, self.ref_body_index],
                        self.robot2.data.joint_pos,
                        self.robot2.data.joint_vel], dim=-1)
        ref2 = torch.cat([self.ref_state_buffer_2["root_state"],
                          self.ref_state_buffer_2["joint_pos"].reshape(self.num_envs, frames, -1),
                          self.ref_state_buffer_2["joint_vel"].reshape(self.num_envs, frames, -1)], dim=-1)
        
        # reward = torch.norm(torch.cat([obs1, obs2], dim=-1) - 
        #                     torch.cat([ref1, ref2], dim=-1)[torch.arange(self.num_envs), self.episode_length_buf], 
        #                     dim=-1)
        reward = torch.mean(torch.cat([obs1, obs2], dim=-1) - 
                            torch.cat([ref1, ref2], dim=-1)[torch.arange(self.num_envs), self.episode_length_buf], 
                            dim=-1)

        return 1.0 - torch.abs(reward)

    def reward_ones(self) -> torch.Tensor:
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
    
    def compute_obs(self,
        dof_positions: torch.Tensor,
        dof_velocities: torch.Tensor,
        root_position: torch.Tensor,
        root_rotation: torch.Tensor,
        root_linear_velocity: torch.Tensor,
        root_angular_velocity: torch.Tensor,
        relative_pose: torch.Tensor | None=None,
    ) -> torch.Tensor:
        obs = torch.cat(
            (
                dof_positions,
                dof_velocities,
                root_position,
                root_rotation,
                root_linear_velocity,
                root_angular_velocity,
            ),
            dim=-1,
        )
        
        if relative_pose is not None: 
            relative_pose = relative_pose.reshape(obs.shape[0], -1)
            obs = torch.cat((obs, relative_pose), dim=-1)
        return obs


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_rotate(q, ref_tangent)
    normal = quat_rotate(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

@torch.jit.script
def check_nan(tensor: torch.Tensor) -> torch.Tensor:
    return torch.isnan(tensor).any(dim=-1)