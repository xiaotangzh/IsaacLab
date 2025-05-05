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
#SBATCH --nodelist=gpu[7]

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
# python play.py --train --task AMP-InterHuman --num_envs 4096 --name "motion26 env4096 steps100w lr1e-3" --steps 1000000 --params 1024 --wandb --headless --disable_progressbar --lr 1e-3

# AMP humanoid
# python play.py --train --task AMP-Humanoid --num_envs 4096 --name "rough walk env4096 steps100w" --steps 1000000 --params 1024 --wandb --headless --disable_progressbar


# AIP
python play.py --train --task AIP-InterHuman-2Robots --num_envs 10000 --name "motion26 env10000 step30w pjd4 GAN_terminate" --steps 300000 --wandb --headless --disable_progressbar

# PPO interhuman
# python play.py --train --task PPO-InterHuman-2Robots --num_envs 512 --name "motion 1 env512 steps 20w imitation" --steps 200000 --wandb --headless --disable_progressbar

# PPO stability
# python play.py --train --task PPO --num_envs 4096 --name "com acc env4096 steps 300000 version 2" --steps 300000 --params 1024 --wandb --headless --disable_progressbar --checkpoint "logs/PPO/com acc env4096 steps 300000/checkpoints/best_agent.pt"

# HRL
# python play.py --train --task HRL-Humanoid --num_envs 1024 --name "walk com_acc steps20w lr1e-3 ini_log_prob-1" --steps 200000 --params 1024 --wandb --headless --disable_progressbar --lr 1e-3 