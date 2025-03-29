import numpy as np 
import pickle
import argparse

def read(in_file):
    # data = np.load("1.npy")
    # print(data.shape)

    with open(in_file, 'rb') as file:
        data = pickle.load(file)
    print(data.keys())
    print(data['person1'].keys())
    print(data['person1']['trans'].shape)
    print(data['person1']['root_orient'].shape)
    print(data['person1']['pose_body'].shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="in_files/1.pkl")
    args = parser.parse_args()
    read(
        in_file=args.in_file,
    )
