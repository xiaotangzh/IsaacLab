#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --qos=long-high-prio 
#SBATCH --time=6-23:59:59
#SBATCH -p res-gpu-small
#SBATCH --mem=10g #max


### 2 x Titan X, 14 x Titan XP
# --gres=gpu:pascal:1          

### 126 x Virtual 10GB GPU (each equivalent to 1/7 compute performance of an 80GB A100)
# --gres=gpu:1g.10gb:1

### 24 x 2080 Ti, 8 x Titan RTX, 1 x Quadro RTX 8000
#SBATCH --gres=gpu:turing:1 

### 3 x RTX A6000, 12 x 80GB A100 (PCIe)
# --gres=gpu:ampere:1       


# Isaac Sim root directory
export ISAACSIM_PATH="/home2/sgtn88/IsaacSim"
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

path_to_your_repo="/home2/sgtn88/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/my_tasks"
export PYTHONPATH=$PYTHONPATH:$path_to_your_repo
cd "/home2/sgtn88/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/my_tasks"

# AMP interhuman
python train.py --task AMP-InterHuman --num_envs 1024 --name "motion26 env1024 steps 50w" --steps 500000 --params 1024 --wandb --headless --disable_progressbar

# AMP humanoid
# python train.py --task AMP-Humanoid --num_envs 1024 --name "dance env1024 steps 50w" --steps 500000 --params 1024 --wandb --headless --disable_progressbar

# PPO interhuman
# python train.py --task PPO-InterHuman-2Robots --num_envs 512 --name "motion 1 env512 steps 20w imitation" --steps 200000 --wandb --headless --disable_progressbar

# PPO stability
# python train.py --task PPO --num_envs 4096 --name "com acc env4096 steps 300000 version 2" --steps 300000 --params 1024 --wandb --headless --disable_progressbar --checkpoint "logs/PPO/com acc env4096 steps 300000/checkpoints/best_agent.pt"

# HRL
# python train.py --task HRL-Humanoid --num_envs 4096 --name "walk com_acc imitation env4096 steps 20w" --steps 200000 --params 1024 --wandb --headless --disable_progressbar