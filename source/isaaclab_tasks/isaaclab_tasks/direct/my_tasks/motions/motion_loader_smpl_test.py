# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
from typing import Optional
from .motion_loader import MotionLoader

class MotionLoaderSMPL(MotionLoader):
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        # assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        # data = np.load(motion_file)

        # self.device = device
        # self._dof_names = data["dof_names"].tolist()
        # self._body_names = data["body_names"].tolist()

        data = super().__init__(motion_file, device)

        self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device)
        self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
        self.root_linear_velocity = torch.tensor(
            data["root_linear_velocity"], dtype=torch.float32, device=self.device
        )
        self.root_angular_velocity = torch.tensor(
            data["root_angular_velocity"], dtype=torch.float32, device=self.device
        )

        # self.dt = 1.0 / data["fps"]
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        # print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

    def sample(
        self, num_samples: int, times: Optional[np.ndarray] = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3)) and body angular velocities (with shape (N, num_bodies, 3)).
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.root_linear_velocity, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.root_angular_velocity, blend=blend, start=index_0, end=index_1),
        )
    
    def get_all_references(self, num_samples: int = 1):
        return (self.dof_positions.clone().unsqueeze(0).expand(num_samples, -1, -1), 
                self.dof_velocities.clone().unsqueeze(0).expand(num_samples, -1, -1),
                self.body_positions.clone().unsqueeze(0).expand(num_samples, -1, -1, -1),
                self.body_rotations.clone().unsqueeze(0).expand(num_samples, -1, -1, -1),
                self.root_linear_velocity.clone().unsqueeze(0).expand(num_samples, -1, -1),
                self.root_angular_velocity.clone().unsqueeze(0).expand(num_samples, -1, -1))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- name of DOFs:", motion._dof_names)
    print("- number of bodies:", motion.num_bodies)
    print("- name of bodies:", motion._body_names)