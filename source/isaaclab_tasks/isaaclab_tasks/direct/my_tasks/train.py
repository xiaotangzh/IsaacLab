from skrl.trainers.torch import SequentialTrainer

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import torch
import argparse
import os
from isaaclab.app import AppLauncher
import gymnasium
import datetime
import sys

from utils.utils import *

# parse the arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=80000, help="Number of training steps.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--name", type=str, default="", help="Name of the experiment.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--wandb", action="store_true", default=False, help="Log training results to Weight and Bias.")
parser.add_argument("--lr", type=float, default=0.0, help="Learning rate.")
parser.add_argument("--params", type=int, default=1024, help="Number of parameters for learning.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load and wrap the Isaac Lab environment
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
experiment_name = f"{args.task} {args.name}" if args.name else f"{args.task} {datetime.datetime.now().strftime('%d_%H-%M')}"

# start the app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# isaacsim.core must be imported after the app is started
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg(args.task, num_envs=args.num_envs)

# wrap environment
env = gymnasium.make(args.task, cfg=cfg, render_mode="rgb_array" if args.video else None)
env = wrap_env(env)

# agent configuration
from agents.amp_2robots import AMP, AMP_DEFAULT_CONFIG
from agents.moe import MOE
from agents.ppo import PPO, PPO_DEFAULT_CONFIG
from models.amp import *
from models.moe import *
from models.ppo import *
agent, agent_cfg = None, None

# IsaacLab AMP default configurations
if "AMP" in args.task:
    agent_cfg = AMP_DEFAULT_CONFIG.copy()
    
    # IsaacLab AMP default configurations
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
    agent_cfg["value_preprocessor"] = RunningStandardScaler 
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1}
    agent_cfg["amp_state_preprocessor"] = RunningStandardScaler
    agent_cfg["amp_state_preprocessor_kwargs"] = {"size": env.amp_observation_size}
    agent_cfg["discriminator_batch_size"] = 4096
    agent_cfg["clip_predicted_values"] = True
    
    # memory configuration
    rollout_memory = RandomMemory(
        memory_size=agent_cfg["rollouts"], 
        num_envs=env.num_envs, 
        device=device  
    )
    motion_dataset = RandomMemory(
        memory_size=200000,
        device=device
    )
    reply_buffer = RandomMemory(
        memory_size=1000000, 
        # num_envs must be 1 to avoid memory samples shape errors
        device=device,
    )
    
    # custom configurations
    if args.lr: agent_cfg["learning_rate"] = args.lr
    agent_cfg["experiment"] = {
        "directory": os.path.join("logs", args.task), 
        "experiment_name": args.name, 
        "write_interval": 100,   
        "checkpoint_interval": "auto",  
        "wandb": args.wandb,      
        "wandb_kwargs": {
            "entity": "xiaotang-zhang",
            "project": "IsaacLab",
            "name": experiment_name,
        }
    }
    
    # instantiate the models
    models = instantiate_AMP_2robots(env, params=args.params, device=device)
    agent = AMP(models=models,
                memory=rollout_memory,  
                cfg=agent_cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                amp_observation_space=env.amp_observation_size,
                motion_dataset=motion_dataset,
                reply_buffer=reply_buffer,
                collect_reference_motions=env.collect_reference_motions,
                device=device)
elif "PPO" in args.task:
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    rollout_memory = RandomMemory(
        memory_size=agent_cfg["rollouts"], 
        num_envs=env.num_envs, 
        device=device  
    )
    
    # custom configurations
    if args.lr: agent_cfg["learning_rate"] = args.lr
    agent_cfg["experiment"] = {
        "directory": os.path.join("logs", args.task), 
        "experiment_name": args.name, 
        "write_interval": 100,   
        "checkpoint_interval": "auto",  
        "wandb": args.wandb,      
        "wandb_kwargs": {
            "entity": "xiaotang-zhang",
            "project": "IsaacLab",
            "name": experiment_name,
        }
    }
    
    models = instantiate_PPO(env, params=args.params, device=device)
    agent = PPO(models=models,
                memory=rollout_memory,  
                cfg=agent_cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": args.steps, "disable_progressbar": True if sys.platform.startswith("Linux") else False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# resume checkpoint (if specified)
if args.checkpoint:
    from isaaclab.utils.assets import retrieve_file_path
    resume_path = retrieve_file_path(args.checkpoint)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)
        # agent.cfg = agent_cfg

# start training
trainer.train()

# close the simulator
env.close()

# close sim app
simulation_app.close()