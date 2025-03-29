import numpy as np

data = np.load("humanoid_dance.npz", allow_pickle=True)
print(data.keys())
print(data['fps'])
print(data['dof_names'])
print(data['body_names'])
print(data['dof_positions'].shape)
print(data['dof_velocities'].shape)
print(data['body_positions'].shape)
print(data['body_rotations'].shape)
print(data['body_linear_velocities'].shape)
print(data['body_angular_velocities'].shape)