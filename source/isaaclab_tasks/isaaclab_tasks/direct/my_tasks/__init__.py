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

gym.register(
    id="My-AMP-InterHuman",
    entry_point=f"{__name__}.my_env:MyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_env_cfg:MyAmpInterHumanEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)

gym.register(
    id="My-PPO",
    entry_point=f"{__name__}.my_env:MyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_env_cfg:MyPPOEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

## Two Robots registrationo
gym.register(
    id="My-AMP-InterHuman-2Robots",
    entry_point=f"{__name__}.my_env_2robots:MyEnv2Robots",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_env_2robots_cfg:MyAmpInterHumanEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)