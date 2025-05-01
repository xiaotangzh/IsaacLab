import time
from skrl.trainers.torch import SequentialTrainer

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import set_seed
from skrl.models.torch import Model
from dataclasses import MISSING

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
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=80000, help="Number of training steps.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--name", type=str, default="", help="Name of the experiment.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--wandb", action="store_true", default=False, help="Log training results to Weight and Bias.")
parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
parser.add_argument("--params", type=int, default=1024, help="Number of parameters for learning.") 
parser.add_argument("--disable_progressbar", action="store_true", default=False, help="Disable progress bar of tqdm.")
parser.add_argument("--train", action="store_true", default=False, help="Training mode.")
parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the models and disable require_grad.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load and wrap the Isaac Lab environment
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
assert args.train ^ args.eval, "Exactly one of --train or --eval must be specified."
assert not (args.eval and args.checkpoint is None), "When --eval is set, --checkpoint must not be None."
experiment_name = f"{args.task} {args.name}" if args.name else f"{args.task} {datetime.datetime.now().strftime('%d_%H-%M')}"
checkpoint_interval = min(30000, args.steps // 10)

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
from agents.amp import AMP, AMP_DEFAULT_CONFIG
from agents.aip import AIP, AIP_DEFAULT_CONFIG
from agents.ppo import PPO, PPO_DEFAULT_CONFIG
from agents.hrl import HRL, HRL_DEFAULT_CONFIG
from models.amp import *
from models.aip import *
from models.hrl import *
from models.ppo import *
agent: BaseAgent = MISSING
agent_cfg: dict = MISSING

if "AMP" in args.task:
    agent_cfg = AMP_DEFAULT_CONFIG.copy()
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1}
    agent_cfg["amp_state_preprocessor_kwargs"] = {"size": env.amp_observation_size}
    agent_cfg["task_reward_weight"] = 1.0
    agent_cfg["learning_rate_scheduler"] = KLAdaptiveLR
    
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
        "checkpoint_interval": checkpoint_interval,  
        "wandb": args.wandb,      
        "wandb_kwargs": {
            "entity": "xiaotang-zhang",
            "project": "IsaacLab",
            "name": experiment_name,
        }
    }
    
    # instantiate the models
    models = instantiate_AMP(env, params=args.params, device=device)
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

if "AIP" in args.task:
    agent_cfg = AIP_DEFAULT_CONFIG.copy()
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1}
    agent_cfg["amp_state_preprocessor_kwargs"] = {"size": env.amp_observation_size}
    agent_cfg["amp_inter_state_preprocessor_kwargs"] = {"size": env.amp_inter_observation_size}
    agent_cfg["task_reward_weight"] = 1.0
    agent_cfg["learning_rate_scheduler"] = KLAdaptiveLR
    
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
    interaction_dataset = RandomMemory(
        memory_size=200000,
        device=device
    )
    reply_buffer_inter = RandomMemory(
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
        "checkpoint_interval": checkpoint_interval,  
        "wandb": args.wandb,      
        "wandb_kwargs": {
            "entity": "xiaotang-zhang",
            "project": "IsaacLab",
            "name": experiment_name,
        }
    }
    
    # instantiate the models
    models = instantiate_AIP(env, params=args.params, device=device)
    agent = AIP(models=models,
                memory=rollout_memory,  
                cfg=agent_cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                amp_observation_space=env.amp_observation_size,
                amp_inter_observation_space=env.amp_inter_observation_size,
                motion_dataset=motion_dataset,
                reply_buffer=reply_buffer,
                interaction_dataset=interaction_dataset,
                reply_buffer_inter=reply_buffer_inter,
                collect_reference_motions=env.collect_reference_motions,
                collect_reference_interactions=env.collect_reference_interactions,
                device=device)
elif "PPO" in args.task:
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    rollout_memory = RandomMemory(
        memory_size=agent_cfg["rollouts"], 
        num_envs=env.num_envs, 
        device=device  
    )

    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
    agent_cfg["value_preprocessor_1"] = RunningStandardScaler 
    agent_cfg["value_preprocessor_2"] = RunningStandardScaler 
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1}
    agent_cfg["clip_predicted_values"] = True
    
    # custom configurations
    if args.lr: agent_cfg["learning_rate"] = args.lr
    agent_cfg["experiment"] = {
        "directory": os.path.join("logs", args.task), 
        "experiment_name": args.name, 
        "write_interval": 100,   
        "checkpoint_interval": checkpoint_interval,  
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
    
elif "HRL" in args.task:
    agent_cfg = HRL_DEFAULT_CONFIG.copy()
    
    # IsaacLab AMP default configurations
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
    agent_cfg["amp_state_preprocessor_kwargs"] = {"size": env.amp_observation_size}
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1}
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
        "checkpoint_interval": checkpoint_interval,  
        "wandb": args.wandb,      
        "wandb_kwargs": {
            "entity": "xiaotang-zhang",
            "project": "IsaacLab",
            "name": experiment_name,
        }
    }

    # load pretrained policy
    checkpoint = "./logs/AMP-Humanoid/PRETRAINED/walk env1024 lr5e-5 steps30w/checkpoints/best_agent.pt"
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    pretrained_policy = instantiate_AMP_policy(env, params=args.params, device=device)
    pretrained_policy.load_state_dict(checkpoint["policy"])
    # load pretrained state preprocessor
    pretrained_state_preprocessor = RunningStandardScaler(**agent_cfg["state_preprocessor_kwargs"])
    pretrained_state_preprocessor.load_state_dict(checkpoint["state_preprocessor"])
    agent_cfg["pretrained_state_preprocessor"] = pretrained_state_preprocessor
    # load pretrained AMP modules
    discriminator = instantiate_AMP_discriminator(env, params=args.params, device=device)
    discriminator.load_state_dict(checkpoint["discriminator"])
    agent_cfg["discriminator"] = discriminator
    amp_state_preprocessor = RunningStandardScaler(**agent_cfg["amp_state_preprocessor_kwargs"])
    amp_state_preprocessor.load_state_dict(checkpoint["amp_state_preprocessor"])
    agent_cfg["amp_state_preprocessor"] = amp_state_preprocessor
    agent_cfg["amp_observation_space"] = env.amp_observation_size
    
    # instantiate the models
    models = instantiate_HRL(env, params=args.params, device=device)
    models["pretrained_policy"] = pretrained_policy

    agent = HRL(models=models,
                memory=rollout_memory,  
                cfg=agent_cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": args.steps, "disable_progressbar": args.disable_progressbar}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# resume checkpoint (if specified)
if args.checkpoint:
    from isaaclab.utils.assets import retrieve_file_path
    resume_path = retrieve_file_path(args.checkpoint)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)

if args.eval: evaluate(agent, env, args)
else: trainer.train()

# close the simulator
env.close()

# close sim app
simulation_app.close()




