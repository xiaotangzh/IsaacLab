# import the agent and its default configuration
from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import torch
from torch import nn
import argparse
import os
from isaaclab.app import AppLauncher
import gymnasium

# parse the arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load and wrap the Isaac Lab environment
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# start the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# importint isaacsim.core must be after the app is started
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)

# wrap environment
env = gymnasium.make(args_cli.task, cfg=cfg, render_mode="rgb_array" if args_cli.video else None)
env = wrap_env(env)

# ==================== Policy Model (Gaussian Policy) ====================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, CLIP=None, device=None):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, 
                              clip_actions=False,
                              clip_log_std=True,
                              min_log_std=-20.0,
                              max_log_std=2.0,
                            #   initial_log_std=-2.9,
                            #   fixed_log_std=True
                              )

        # Network layers (1024 -> 512)
        self.net = nn.Sequential(
            nn.Linear(observation_space, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)  # Output actions
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space)) #todo different from default yaml

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

# ==================== Value Model (Deterministic) ====================
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device=None):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # Network layers (1024 -> 512 -> 1)
        self.net = nn.Sequential(
            nn.Linear(observation_space, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Output single value
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        return self.net(states), {}  # (value, None)

# ==================== Discriminator Model (Deterministic) ====================
class Discriminator(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device=None):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # Network layers (same as Value network)
        self.net = nn.Sequential(
            nn.Linear(observation_space, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Output single discriminator score
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        return self.net(states), {}  # (discriminator_output, None)
    
# instantiate the agent's models
models = {}
models["policy"] = Policy(env.observation_space.shape[0], env.action_space.shape[0], device)
models["value"] = Value(env.observation_space.shape[0], env.action_space.shape[0], device)
models["discriminator"] = Discriminator(env.amp_observation_size, env.action_space.shape[0], device)

# adjust some configuration if necessary
amp_cfg = AMP_DEFAULT_CONFIG.copy()

# =============== 训练参数 ===============
amp_cfg["rollouts"] = 16                     # 每次 rollout 的步数
amp_cfg["learning_epochs"] = 6               # 每次更新的训练轮次
amp_cfg["mini_batches"] = 2                  # 每轮的 mini-batch 数量
amp_cfg["discount_factor"] = 0.99            # 折扣因子 γ
amp_cfg["lambda"] = 0.95                     # GAE 参数 λ
amp_cfg["learning_rate"] = 5.0e-05           # 学习率

# =============== 预处理 ===============
amp_cfg["state_preprocessor"] = RunningStandardScaler    # 状态标准化
amp_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
amp_cfg["value_preprocessor"] = RunningStandardScaler    # 值函数标准化
amp_cfg["value_preprocessor_kwargs"] = {"size": 1}
amp_cfg["amp_state_preprocessor"] = RunningStandardScaler  # AMP 状态标准化
amp_cfg["amp_state_preprocessor_kwargs"] = {"size": env.amp_observation_size}

# =============== 损失函数缩放 ===============
amp_cfg["value_loss_scale"] = 2.5            # 值函数损失权重
amp_cfg["discriminator_loss_scale"] = 5.0    # 判别器损失权重
amp_cfg["entropy_loss_scale"] = 0.0          # 熵正则化权重 (0表示禁用)

# =============== AMP 专用参数 ===============
amp_cfg["amp_batch_size"] = 512               # AMP 参考运动数据的 batch 大小
amp_cfg["task_reward_weight"] = 0.0           # 任务奖励权重 (0表示仅用风格奖励)
amp_cfg["style_reward_weight"] = 1.0          # 风格奖励权重
amp_cfg["discriminator_batch_size"] = 4096    # 判别器 batch 大小
amp_cfg["discriminator_reward_scale"] = 2.0   # 判别器奖励缩放因子
amp_cfg["discriminator_logit_regularization_scale"] = 0.05  # 判别器 logit 正则化
amp_cfg["discriminator_gradient_penalty_scale"] = 5.0      # 梯度惩罚系数
amp_cfg["discriminator_weight_decay_scale"] = 1.0e-04      # L2 正则化系数

# =============== 其他参数 ===============
amp_cfg["grad_norm_clip"] = 0.0              # 梯度裁剪 (0表示禁用)
amp_cfg["ratio_clip"] = 0.2                  # PPO clip 参数
amp_cfg["value_clip"] = 0.2                  # 值函数 clip 参数
amp_cfg["clip_predicted_values"] = True      # 是否裁剪预测值
amp_cfg["random_timesteps"] = 0              # 随机探索步数
amp_cfg["learning_starts"] = 0               # 开始学习前的步数
amp_cfg["time_limit_bootstrap"] = False      # 是否对超时终止做 bootstrap

# =============== 日志与检查点 ===============
amp_cfg["experiment"] = {
    "directory": os.path.join("logs", "skrl", "humanoid_amp_interhuman"),  # 实验目录
    "experiment_name": "",                   # 实验名称 (可选)
    "write_interval": "auto",                # 日志写入间隔
    "checkpoint_interval": "auto"            # 模型保存间隔
}

# ==================== Rollout Memory ====================
# 自动根据 agent 的 rollouts 步数确定大小
rollout_memory = RandomMemory(
    memory_size=amp_cfg["rollouts"],  # 自动计算（与 agent.rollouts 相同）
    num_envs=env.num_envs, 
    device=device  
)

# ==================== AMP Motion Dataset (Reference Motion) ====================
motion_dataset = RandomMemory(
    memory_size=200000,
    # num_envs=1,  # 通常不与环境数量关联
    device=device
)

# ==================== Replay Buffer (防止 Discriminator 过拟合) ====================
reply_buffer = RandomMemory(
    memory_size=1000000, 
    # num_envs=env.num_envs,
    device=device,
)

agent = AMP(models=models,
            memory=rollout_memory,  
            cfg=amp_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            amp_observation_space=env.amp_observation_size,
            motion_dataset=motion_dataset,
            reply_buffer=reply_buffer,
            collect_reference_motions=env.collect_reference_motions,
            device=device)

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 80000}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# resume checkpoint (if specified)
if args_cli.checkpoint:
    from isaaclab.utils.assets import retrieve_file_path
    resume_path = retrieve_file_path(args_cli.checkpoint)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)

# start training
trainer.train()