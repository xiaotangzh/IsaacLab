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

path_to_your_repo="/home2/sgtn88/IsaacSim"
export PYTHONPATH=$PYTHONPATH:$path_to_your_repo
cd ..

# note: you can pass the argument "--help" to see all arguments possible.
# ${ISAACSIM_PATH}/isaac-sim.sh

# checks that python path is set correctly
# ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"

# checks that Isaac Sim can be launched from python
${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.api/add_cubes.py