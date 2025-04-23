# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


## AMP
gym.register(
    id="AMP-InterHuman-2Robots",
    entry_point=f"{__name__}.env:Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_env_cfg:Amp_InterHuman_2Robots",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)
gym.register(
    id="AMP-InterHuman",
    entry_point=f"{__name__}.env:Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_env_cfg:Amp_InterHuman",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)

### PPO
gym.register(
    id="PPO-InterHuman",
    entry_point=f"{__name__}.env:Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_env_cfg:PPO_InterHuman",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="PPO-Humanoid",
    entry_point=f"{__name__}.env:Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_env_cfg:PPO_Humanoid",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="PPO-InterHuman-2Robots",
    entry_point=f"{__name__}.env:Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_env_cfg:PPO_InterHuman_2Robots",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


### HRL
gym.register(
    id="HRL-InterHuman",
    entry_point=f"{__name__}.env:Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_env_cfg:HRL_InterHuman",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)
gym.register(
    id="HRL-Humanoid",
    entry_point=f"{__name__}.env:Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_env_cfg:HRL_Humanoid",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)