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

from .env_1robot_cfg import Env1RobotCfg
from .motions.motion_loader import MotionLoader
import sys

class Env1Robot(DirectRLEnv):
    cfg: Env1RobotCfg

    def __init__(self, cfg: Env1RobotCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        self.init_root_height = 0.07 # lift the humanoid slightly to avoid collisions with the ground
        self.termination_heights = torch.tensor(self.cfg.termination_heights, device=self.device)

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # DOF and key body indexes
        key_body_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe", "Head"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.early_termination_body_indexes = [self.robot.data.body_names.index(name) for name in self.cfg.termination_bodies]
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names) # self.robot.data.joint_names: 'L_Hip_x', 'L_Hip_y' ...
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )
        
        # sync motion
        self.ref_state_buffer_length, self.ref_state_buffer_index = 400, 0
        self.ref_state_buffer = {}
        if self.cfg.sync_motion:
            self.reset_reference_buffer()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
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
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    #! Pre-physics step
    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.clip(actions, min=-0.07, max=0.07) # clip the actions
        if torch.isnan(actions).any():
            print("nan in actions")
            sys.exit(0)
        self.actions = actions.clone()

    def _apply_action(self):
        if self.cfg.sync_motion:
            self.write_ref_state()
        else:
            target = self.action_offset + self.action_scale * self.actions
            self.robot.set_joint_position_target(target)
    #! Pre-physics step (End)

    #! Post-physics step
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: # should return resets and time_out
        time_out = self.episode_length_buf >= self.max_episode_length - 1 # bools of envs that are time out
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.early_termination_body_indexes, 2] < self.termination_heights
            died = torch.max(died, dim=1).values
        else:
            died = torch.zeros_like(time_out) # no early termination until time out
        
        # end of reference buffer
        if self.cfg.sync_motion and self.ref_state_buffer_index >= self.ref_state_buffer_length:
            died = torch.ones_like(time_out)
            
        return died, time_out
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)

    def _reset_idx(self, env_ids: torch.Tensor | None): # env_ids: the ids of envs to be reset
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self.reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self.reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")
        
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
    def _get_observations(self) -> dict:
        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
        )

        if torch.isnan(obs).any():
            print("NaN in observation, stop training.")
            sys.exit(0)

        # update AMP observation history (pop)
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation (push)
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}
    #! Post-physics step (End)
    
    def write_ref_state(self):
        self.robot.write_root_link_pose_to_sim(self.ref_state_buffer['root_state'][:, self.ref_state_buffer_index, :7], 
                                               self.robot._ALL_INDICES)
        self.robot.write_root_com_velocity_to_sim(self.ref_state_buffer['root_state'][:, self.ref_state_buffer_index, 7:], 
                                                  self.robot._ALL_INDICES)
        
        #todo what is the difference between the two lines below?
        # self.robot.write_root_pose_to_sim(self.ref_state_buffer['root_state'][:, self.ref_state_buffer_index, :7], 
        #                                        self.robot._ALL_INDICES)
        # self.robot.write_root_state_to_sim(self.ref_state_buffer['root_state'][:, self.ref_state_buffer_index], 
        #                                           self.robot._ALL_INDICES)
        
        self.robot.write_joint_state_to_sim(self.ref_state_buffer['joint_pos'][:, self.ref_state_buffer_index], 
                                            self.ref_state_buffer['joint_vel'][:, self.ref_state_buffer_index], 
                                            None, self.robot._ALL_INDICES)
        
        self.ref_state_buffer_index = (self.ref_state_buffer_index + 1) % self.ref_state_buffer_length

    # reset strategies
    def reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.1  # lift the humanoid slightly to avoid collisions with the ground
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # env_ids: the ids of envs to be reset
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            root_linear_velocity,
            root_angular_velocity,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        
        # get root transforms (the humanoid torso)
        motion_root_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_root_index] + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = body_rotations[:, motion_root_index]
        root_state[:, 7:10] = root_linear_velocity
        root_state[:, 10:13] = root_angular_velocity
        root_state[:, 2] += self.init_root_height  # lift the humanoid slightly to avoid collisions with the ground
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)
        
        return root_state, dof_pos, dof_vel
    
    def reset_reference_buffer(self, env_ids: torch.Tensor | None=None):
        env_ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        num_samples = env_ids.shape[0]
        motion_root_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        
        # sample reference actions
        (
            ref_dof_positions,
            ref_dof_velocities,
            ref_body_positions,
            ref_body_rotations,
            ref_root_linear_velocity,
            ref_root_angular_velocity,
        ) = self._motion_loader.get_all_references(num_samples)
        ref_root_state = self.robot.data.default_root_state[env_ids].unsqueeze(1).expand(-1, ref_dof_positions.shape[1], -1).clone()
        ref_root_state[:, :, 0:3] = ref_body_positions[:, :, motion_root_index] + self.scene.env_origins[env_ids].unsqueeze(1)
        ref_root_state[:, :, 2] += self.init_root_height  # lift the humanoid slightly to avoid collisions with the ground
        ref_root_state[:, :, 3:7] = ref_body_rotations[:, :, motion_root_index]
        ref_root_state[:, :, 7:10] = ref_root_linear_velocity
        ref_root_state[:, :, 10:13] = ref_root_angular_velocity
        self.ref_state_buffer = {
            "root_state": ref_root_state,
            "joint_pos": ref_dof_positions[:, :, self.motion_dof_indexes],
            "joint_vel": ref_dof_velocities[:, :, self.motion_dof_indexes],
        }

    # env.wrapper -> (skrl Runner) -> skrl AMP agent
    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            root_linear_velocity,
            root_angular_velocity,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # self.extras["text"] = #todo pass sampled motion text description to env.infos
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            root_linear_velocity,
            root_angular_velocity,
        )
        return amp_observation.view(-1, self.amp_observation_size) # (num_envs, state transitions)


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
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_position: torch.Tensor,
    root_rotation: torch.Tensor,
    root_linear_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
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
    return obs
