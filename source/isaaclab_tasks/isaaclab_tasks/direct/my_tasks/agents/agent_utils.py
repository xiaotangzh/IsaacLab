import re
import torch
import torch.nn.functional as F
import torch.nn as nn
from skrl.models.torch import Model
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_tasks.direct.my_tasks.agents.base_agent import BaseAgent  # avoid circular import error

def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Generalized Advantage Estimator (GAE)

    :param rewards: Rewards obtained by the agent
    :type rewards: torch.Tensor
    :param dones: Signals to indicate that episodes have ended
    :type dones: torch.Tensor
    :param values: Values obtained by the agent
    :type values: torch.Tensor
    :param next_values: Next values obtained by the agent
    :type next_values: torch.Tensor
    :param discount_factor: Discount factor
    :type discount_factor: float
    :param lambda_coefficient: Lambda coefficient
    :type lambda_coefficient: float

    :return: Generalized Advantage Estimator
    :rtype: torch.Tensor
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_dones = dones.logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        advantage = (
            rewards[i]
            - values[i]
            + discount_factor * (next_values[i] + lambda_coefficient * not_dones[i] * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

def compute_policy_loss(
    agent: "BaseAgent",
    next_log_prob: torch.Tensor,
    sampled_log_prob: torch.Tensor,
    sampled_advantages: torch.Tensor
) -> torch.Tensor:
    ratio = torch.exp(next_log_prob - sampled_log_prob)
    surrogate = sampled_advantages * ratio
    surrogate_clipped = sampled_advantages * torch.clip(
        ratio, 1.0 - agent._ratio_clip, 1.0 + agent._ratio_clip
    )

    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
    return policy_loss

def compute_value_loss(
    agent: "BaseAgent",
    value: "Model",
    sampled_states: torch.Tensor,
    sampled_values: torch.Tensor,
    sampled_returns: torch.Tensor,
) -> torch.Tensor:
    predicted_values, _, _ = value.act({"states": sampled_states}, role="value")

    if agent._clip_predicted_values:
        predicted_values = sampled_values + torch.clip(
            predicted_values - sampled_values, min=-agent._value_clip, max=agent._value_clip
        )
    value_loss = agent._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
    return value_loss

def compute_discriminator_loss(
    agent: "BaseAgent",
    discriminator: "Model",
    amp_state_preprocessor,
    batch_index,
    sampled_amp_states: torch.Tensor,
    sampled_replay_batches: list,
    sampled_motion_batches: list,
    weight: float = 1.0
) -> torch.Tensor:
    # compute discriminator loss
    if agent._discriminator_batch_size:
        sampled_amp_states = amp_state_preprocessor(
            sampled_amp_states[0 : agent._discriminator_batch_size], train=True
        )
        sampled_amp_replay_states = amp_state_preprocessor(
            sampled_replay_batches[batch_index][0][0 : agent._discriminator_batch_size], train=True
        )
        sampled_amp_motion_states = amp_state_preprocessor(
            sampled_motion_batches[batch_index][0][0 : agent._discriminator_batch_size], train=True
        )
    else:
        sampled_amp_states = amp_state_preprocessor(sampled_amp_states, train=True)
        sampled_amp_replay_states = amp_state_preprocessor(
            sampled_replay_batches[batch_index][0], train=True
        )
        sampled_amp_motion_states = amp_state_preprocessor(
            sampled_motion_batches[batch_index][0], train=True
        )

    sampled_amp_motion_states.requires_grad_(True)
    amp_logits, _, _ = discriminator.act({"states": sampled_amp_states}, role="discriminator")
    amp_replay_logits, _, _ = discriminator.act(
        {"states": sampled_amp_replay_states}, role="discriminator"
    )
    amp_motion_logits, _, _ = discriminator.act(
        {"states": sampled_amp_motion_states}, role="discriminator"
    )

    amp_cat_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)

    # discriminator prediction loss
    discriminator_loss = 0.5 * (
        nn.BCEWithLogitsLoss()(amp_cat_logits, torch.zeros_like(amp_cat_logits))
        + torch.nn.BCEWithLogitsLoss()(amp_motion_logits, torch.ones_like(amp_motion_logits))
    ) * weight

    # discriminator logit regularization
    if agent._discriminator_logit_regularization_scale:
        logit_weights = torch.flatten(list(discriminator.modules())[-1].weight)
        discriminator_loss += agent._discriminator_logit_regularization_scale * torch.sum(
            torch.square(logit_weights)
        )

    # discriminator gradient penalty
    if agent._discriminator_gradient_penalty_scale:
        amp_motion_gradient = torch.autograd.grad(
            amp_motion_logits,
            sampled_amp_motion_states,
            grad_outputs=torch.ones_like(amp_motion_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
        discriminator_loss += agent._discriminator_gradient_penalty_scale * gradient_penalty

    # discriminator weight decay
    if agent._discriminator_weight_decay_scale:
        weights = [
            torch.flatten(module.weight)
            for module in discriminator.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
        discriminator_loss += agent._discriminator_weight_decay_scale * weight_decay

    discriminator_loss *= agent._discriminator_loss_scale
    return discriminator_loss

def compute_entropy_loss(
    agent: "BaseAgent",
    policy: "Model",
):
    if agent._entropy_loss_scale:
        entropy_loss = agent._entropy_loss_scale * policy.get_entropy(role="policy").mean()
    else:
        entropy_loss = 0
    return entropy_loss

def compute_kl(
    next_log_prob: torch.Tensor,
    sampled_log_prob: torch.Tensor,
):
    with torch.no_grad():
        ratio = next_log_prob - sampled_log_prob
        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
    return kl_divergence