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
        "env_cfg_entry_point": f"{__name__}.task_cfg:AmpInterHumanEnvCfg2Robots",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)
gym.register(
    id="AMP-InterHuman",
    entry_point=f"{__name__}.env:Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_cfg:AmpInterHumanEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)

### PPO
gym.register(
    id="PPO",
    entry_point=f"{__name__}.env:Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_cfg:PPOEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="PPOHumanoid",
    entry_point=f"{__name__}.env:Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_cfg:PPOHumanoidEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

