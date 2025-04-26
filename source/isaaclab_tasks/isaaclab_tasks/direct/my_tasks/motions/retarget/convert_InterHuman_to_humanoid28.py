from SkeletonMotion_to_MotionLoader import SkeletonMotion_to_MotionLoader
import numpy as np
from visualization import animate3D

motion = SkeletonMotion_to_MotionLoader(path = "./out_files/Humanoid28_SKMotion_InterHuman_1_person1.npy")
print(motion.keys())
animate3D(motion['body_positions'], highlight_joint=0, q=motion['body_rotations'][:,0], w_last=False)

np.savez("../InterHuman_humanoid28/1_person1.npz", **motion)

# Check the saved file
data = np.load("../InterHuman_humanoid28/1_person1.npz")
for key in data.keys():
    print(key)