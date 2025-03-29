import numpy as np

data = np.load("1.npy", allow_pickle=True)
print(data.item(0).keys())
print('rotation', data.item(0)['rotation']['arr'].shape)
print('root translation', data.item(0)['root_translation']['arr'].shape)
print('global vel', data.item(0)['global_velocity']['arr'].shape)
print('global angular vel', data.item(0)['global_angular_velocity']['arr'].shape)
print('skeleton tree', data.item(0)['skeleton_tree']['node_names'])
print('skeleton tree', data.item(0)['skeleton_tree']['local_translation']['arr'].shape)
print(data.item(0)['is_local'])
print(data.item(0)['fps'])