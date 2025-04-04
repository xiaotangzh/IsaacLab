import numpy as np
from motion_lib import animate3D
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../InterHuman/1.npz")
    args = parser.parse_args()
    data = np.load(args.file)
    
    animate3D(data['body_positions'], highlight_joint=0, q=data['body_rotations'][:,0], w_last=False)