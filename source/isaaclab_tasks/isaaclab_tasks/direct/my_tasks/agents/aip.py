from code import interact
from operator import is_
from random import sample
import sys
from turtle import st
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import copy
import itertools
import math
from click import style
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger

from .base_agent import BaseAgent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.resources.preprocessors.torch import RunningStandardScaler
from isaaclab_tasks.direct.my_tasks.utils import *
from isaaclab_tasks.direct.my_tasks.utils.agent_utils import *
from isaaclab_tasks.direct.my_tasks.bridge.bridge import Bridge

# fmt: off
# [start-config-dict-torch]
AIP_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 6,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-4,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": RunningStandardScaler,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": RunningStandardScaler,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})
    "amp_state_preprocessor": RunningStandardScaler,         # AMP state preprocessor class (see skrl.resources.preprocessors)
    "amp_state_preprocessor_kwargs": {},    # AMP state preprocessor's kwargs (e.g. {"size": env.amp_observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.0,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": True,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,          # entropy loss scaling factor
    "value_loss_scale": 2.5,            # value loss scaling factor
    "discriminator_loss_scale": 5.0,    # discriminator loss scaling factor

    "amp_batch_size": 512,                  # batch size for updating the reference motion dataset
    "task_reward_weight": 0.0,              # task-reward weight (wG)
    "style_reward_weight": 1.0,             # style-reward weight (wS)
    "discriminator_batch_size": 4096,          # batch size for computing the discriminator loss (all samples if 0)
    "discriminator_reward_scale": 2,                    # discriminator reward scaling factor
    "discriminator_logit_regularization_scale": 0.05,   # logit regularization scale factor for the discriminator loss
    "discriminator_gradient_penalty_scale": 5,          # gradient penalty scaling factor for the discriminator loss
    "discriminator_weight_decay_scale": 0.0001,         # weight decay scaling factor for the discriminator loss

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class AIP(BaseAgent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        amp_observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        amp_inter_observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        motion_dataset: Optional[Memory] = None,
        reply_buffer: Optional[Memory] = None,
        interaction_dataset: Optional[Memory] = None,
        reply_buffer_inter: Optional[Memory] = None,
        collect_reference_motions: Optional[Callable[[int], torch.Tensor]] = None,
        collect_reference_interactions: Optional[Callable[[int], torch.Tensor]] = None,
        collect_observation: Optional[Callable[[], torch.Tensor]] = None,
        bridge: Bridge | None = None,
    ) -> None:
        
        _cfg = copy.deepcopy(AIP_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        self.amp_observation_space = amp_observation_space
        self.motion_dataset = motion_dataset
        self.amp_inter_observation_space = amp_inter_observation_space
        self.interaction_dataset = interaction_dataset
        self.reply_buffer = reply_buffer
        self.reply_buffer_inter = reply_buffer_inter
        self.collect_reference_motions = collect_reference_motions
        self.collect_reference_interactions = collect_reference_interactions
        self.collect_observation = collect_observation
        self.bridge = bridge

        # models
        self.policy = self.models.get("policy")
        self.value = self.models.get("value")
        self.discriminator = self.models.get("discriminator")
        self.inter_discriminator = self.models.get("interaction discriminator")

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value
        self.checkpoint_modules["discriminator"] = self.discriminator
        self.checkpoint_modules["interaction discriminator"] = self.inter_discriminator

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]
        self._discriminator_loss_scale = self.cfg["discriminator_loss_scale"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]
        self._amp_state_preprocessor = self.cfg["amp_state_preprocessor"]
        self._amp_inter_state_preprocessor = self.cfg["amp_state_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._amp_batch_size = self.cfg["amp_batch_size"]
        self._task_reward_weight = self.cfg["task_reward_weight"]
        self._style_reward_weight = self.cfg["style_reward_weight"]

        self._discriminator_batch_size = self.cfg["discriminator_batch_size"]
        self._discriminator_reward_scale = self.cfg["discriminator_reward_scale"]
        self._discriminator_logit_regularization_scale = self.cfg["discriminator_logit_regularization_scale"]
        self._discriminator_gradient_penalty_scale = self.cfg["discriminator_gradient_penalty_scale"]
        self._discriminator_weight_decay_scale = self.cfg["discriminator_weight_decay_scale"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None and self.discriminator is not None:
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.policy.parameters(), self.value.parameters(), self.discriminator.parameters(), self.inter_discriminator.parameters()),
                lr=self._learning_rate,
            )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

        if self._amp_state_preprocessor:
            self._amp_state_preprocessor = self._amp_state_preprocessor(**self.cfg["amp_state_preprocessor_kwargs"])
            self.checkpoint_modules["amp_state_preprocessor"] = self._amp_state_preprocessor

            self._amp_inter_state_preprocessor = self._amp_inter_state_preprocessor(**self.cfg["amp_inter_state_preprocessor_kwargs"])
            self.checkpoint_modules["amp_interaction_preprocessor"] = self._amp_inter_state_preprocessor
        else:
            self._amp_state_preprocessor = self._empty_preprocessor
            self._amp_inter_state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
            
            self.memory.create_tensor(name="amp_states", size=self.amp_observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="amp_inter_states", size=self.amp_inter_observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="interaction_reward_weights", size=1, dtype=torch.float32)

        self.tensors_names = [
            "states",
            "actions",
            "rewards",
            "next_states",
            "terminated",
            "log_prob",
            "values",
            "returns",
            "advantages",
            "amp_states",
            "amp_inter_states",
            "next_values",
            "interaction_reward_weights"
        ]

        # create tensors for motion dataset and reply buffer
        if self.motion_dataset is not None:
            self.motion_dataset.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)
            self.interaction_dataset.create_tensor(name="states", size=self.amp_inter_observation_space, dtype=torch.float32)
            self.reply_buffer.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)
            self.reply_buffer_inter.create_tensor(name="states", size=self.amp_inter_observation_space, dtype=torch.float32)

            # initialize motion dataset
            for _ in range(math.ceil(self.motion_dataset.memory_size / self._amp_batch_size)):
                self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))
                self.interaction_dataset.add_samples(states=self.collect_reference_interactions(self._amp_batch_size))

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        # use collected states
        if self._current_states is not None:
            states = self._current_states

        states = self._state_preprocessor(states)

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act({"states": states}, role="policy")
            self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        # use collected states
        if self._current_states is not None:
            states = self._current_states

        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            amp_states = infos["amp_obs"]
            amp_inter_states = infos["amp_interaction_obs"]
            interaction_reward_weights = infos["interaction_reward_weights"]

            # if two robots
            actual_num_envs = rewards.shape[0]
            if states.shape[0] != rewards.shape[0]:
                rewards = rewards.repeat(2, 1) #todo: 2 robot task rewards can be different, replace repeat with expand
                truncated = truncated.repeat(2, 1)
                interaction_reward_weights = interaction_reward_weights.repeat(2, 1)

                # 2 robot termination should be different
                terminated = torch.cat([infos["terminated_1"].view(actual_num_envs, 1), 
                                        infos["terminated_2"].view(actual_num_envs, 1)], dim=0) # [2 * actual_num_envs, 1]

            # test: early termination from discriminator
            style_loss = compute_discriminator_loss(self, self.discriminator, self._amp_state_preprocessor, amp_states).view(actual_num_envs, -1, 1)
            interaction_loss = compute_discriminator_loss(self, self.inter_discriminator, self._amp_inter_state_preprocessor, amp_inter_states).view(actual_num_envs, -1, 1)
            self.track_data("Loss / Style loss", torch.mean(style_loss).item())
            self.track_data("Loss / Interaction loss", torch.mean(interaction_loss).item())
            # if not timestep % 30:
            #     # loss 1.5 ~ sigmoid 0.25
            #     # loss 2.0 ~ signoid 0.15
            #     # loss 3.0 ~ sigmoid 0.05
            #     if torch.mean(style_loss).item() > 2.0: # only if motion quality is too bad (local optima)
            #         style_terminates = (style_loss > 2.0).any(dim=-1) # [actual_num_envs, 1 or 2]
            #         self.bridge.set_terminates(style_terminates)

                # if torch.mean(style_loss) > 1.2: # focus on basic motion style
                    # terminates = style_terminates.clone()
                # else: # focus on interaction quality
                #     interaction_terminates = (interaction_loss > 1.5).any(dim=1).squeeze() 
                #     terminates = torch.max(torch.stack([style_terminates, interaction_terminates], dim=0), dim=0).values
                # self.bridge.set_terminates(terminates)
            
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping 
            # todo: look into if we need this
            # enabling this will make the reward estimation more accurate and long-term
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # compute next values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                next_values, _, _ = self.value.act({"states": self._state_preprocessor(next_states)}, role="value")
                next_values = self._value_preprocessor(next_values, inverse=True)
                next_values *= terminated.view(-1, 1).logical_not()

            # print(states.shape,
            #     actions.shape,
            #     rewards.shape,
            #     next_states.shape,
            #     terminated.shape,
            #     truncated.shape,
            #     self._current_log_prob.shape,
            #     values.shape,
            #     amp_states.shape,
            #     amp_inter_states.shape,
            #     next_values.shape,
            #     interaction_reward_weights.shape,)

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
                amp_states=amp_states,
                amp_inter_states=amp_inter_states,
                next_values=next_values,
                interaction_reward_weights=interaction_reward_weights,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                    amp_states=amp_states,
                    amp_inter_states=amp_inter_states,
                    next_values=next_values,
                    interaction_reward_weights=interaction_reward_weights,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.collect_observation is not None:
            self._current_states = self.collect_observation()

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        if self.bridge: self.bridge.set_timestep(timestep, timesteps)

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # update dataset of reference motions
        self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))
        self.interaction_dataset.add_samples(states=self.collect_reference_interactions(self._amp_batch_size))

        # get data from memory
        task_rewards = self.memory.get_tensor_by_name("rewards")
        amp_states = self.memory.get_tensor_by_name("amp_states")
        amp_inter_states = self.memory.get_tensor_by_name("amp_inter_states")
        interaction_reward_weights = self.memory.get_tensor_by_name("interaction_reward_weights")

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            style_reward = compute_discriminator_reward(self, self.discriminator, self._amp_state_preprocessor, amp_states, task_rewards.shape)
            interaction_reward = compute_discriminator_reward(self, self.inter_discriminator, self._amp_inter_state_preprocessor, amp_inter_states, task_rewards.shape)

        task_rewards = task_rewards * self._task_reward_weight # 1 or 0
        style_reward = style_reward * 3 #* interaction_reward_weights
        interaction_reward = interaction_reward #* interaction_reward_weights
        combined_rewards = task_rewards + style_reward + interaction_reward 

        # log discriminator rewards
        self.track_data("Reward / Style reward", torch.mean(style_reward).item())
        self.track_data("Reward / Interaction reward", torch.mean(interaction_reward).item())

        # compute returns and advantages
        values = self.memory.get_tensor_by_name("values")
        next_values = self.memory.get_tensor_by_name("next_values")
        returns, advantages = compute_gae(
            rewards=combined_rewards,
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=next_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True)) 
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = sample_mini_batches(self, self.tensors_names)
        sampled_motion_batches, sampled_replay_batches = sample_mini_batches_for_discriminator(self, self.tensors_names, self.motion_dataset, self.reply_buffer, sampled_batches, "amp_states")
        sampled_interaction_batches, sampled_replay_inter_batches = sample_mini_batches_for_discriminator(self, self.tensors_names, self.interaction_dataset, self.reply_buffer_inter, sampled_batches, "amp_inter_states")

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_discriminator_loss = 0
        cumulative_inter_discriminator_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for batch_index, (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_amp_states,
                sampled_amp_inter_states,
                sampled_next_values,
                interaction_reward_weights
            ) in enumerate(sampled_batches):

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=True)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    # compute approximate KL divergence
                    kl_divergences.append(compute_kl(next_log_prob, sampled_log_prob))

                    # compute entropy loss
                    entropy_loss = compute_entropy_loss(self, self.policy)

                    # compute policy loss
                    policy_loss = compute_policy_loss(self, next_log_prob, sampled_log_prob, sampled_advantages)

                    # compute value loss
                    value_loss = compute_value_loss(self, self.value, sampled_states, sampled_values, sampled_returns)

                    # compute discriminator loss
                    discriminator_loss = compute_batch_discriminator_loss(self, self.discriminator, self._amp_state_preprocessor, batch_index, sampled_amp_states, sampled_replay_batches, sampled_motion_batches)
                    inter_discriminator_loss = compute_batch_discriminator_loss(self, self.inter_discriminator, self._amp_inter_state_preprocessor, batch_index, sampled_amp_inter_states, sampled_replay_inter_batches, sampled_interaction_batches)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss + discriminator_loss + inter_discriminator_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                cumulative_discriminator_loss += discriminator_loss.item()
                cumulative_inter_discriminator_loss += inter_discriminator_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                update_learning_rate(self, self.scheduler, kl_divergences, config)

        # update AMP replay buffer
        self.reply_buffer.add_samples(states=amp_states.view(-1, amp_states.shape[-1]))
        self.reply_buffer_inter.add_samples(states=amp_inter_states.view(-1, amp_inter_states.shape[-1]))

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )
        self.track_data(
            "Loss / Discriminator loss", cumulative_discriminator_loss / (self._learning_epochs * self._mini_batches)
        )
        self.track_data(
            "Loss / Interaction discriminator loss",
            cumulative_inter_discriminator_loss / (self._learning_epochs * self._mini_batches),
        )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
