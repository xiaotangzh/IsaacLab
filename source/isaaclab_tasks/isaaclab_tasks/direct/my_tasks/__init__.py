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
    id="AMP-InterHuman",
    entry_point=f"{__name__}.env_1robot:Env1Robot",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_1robot_cfg:AmpInterHumanEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)

gym.register(
    id="PPO",
    entry_point=f"{__name__}.env_1robot:Env1Robot",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_1robot_cfg:PPOEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

## Two Robots registrationo
gym.register(
    id="AMP-InterHuman-2Robots",
    entry_point=f"{__name__}.env_2robots:Env2Robots",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_2robots_cfg:AmpInterHumanEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)

gym.register(
    id="PPO-InterHuman-2Robots",
    entry_point=f"{__name__}.env_2robots:Env2Robots",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_2robots_cfg:PPOInterHumanEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_interhuman_cfg.yaml",
    },
)