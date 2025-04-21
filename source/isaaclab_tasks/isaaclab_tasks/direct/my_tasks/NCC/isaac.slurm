#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --qos=long-high-prio 
#SBATCH --time=6-23:59:59
#SBATCH -p res-gpu-small
#SBATCH --mem=28g #max

#* gpu memory=10gb
# --gres=gpu:pascal:1
# --gres=gpu:1g.10gb:1
# --gres=gpu:turing:1

#* gpu78
#SBATCH --gres=gpu:turing:1
#SBATCH --nodelist=gpu[7]

#* A100
# --gres=gpu:ampere:1

# Isaac Sim root directory
export ISAACSIM_PATH="/home2/sgtn88/IsaacSim"
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

path_to_your_repo="/home2/sgtn88/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/my_tasks"
export PYTHONPATH=$PYTHONPATH:$path_to_your_repo
cd "/home2/sgtn88/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/my_tasks"

# AMP interhuman
# python train.py --task AMP-InterHuman-2Robots --num_envs 4096 --name "motion 1 env4096 steps 8w require another pose" --steps 80000 --params 1024 --wandb --headless --disable_progressbar

# PPO interhuman
python train.py --task PPO-InterHuman-2Robots --num_envs 4096 --name "motion 1 env4096 steps 8w" --steps 80000 -wandb --headless --disable_progressbar

# PPO stability
# python train.py --task PPO --num_envs 4096 --name "com acc env4096 steps 300000 version 2" --steps 300000 --params 1024 --wandb --headless --disable_progressbar --checkpoint "logs/PPO/com acc env4096 steps 300000/checkpoints/best_agent.pt"

# HRL
# python train.py --task HRL-Humanoid --num_envs 4096 --name "run flat terrain 0.15 env4096 steps 20w" --steps 200000 --params 1024 --wandb --headless --disable_progressbar