# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import agents
from . import task_env_cfg

for name, cls in task_env_cfg.__dict__.items():
    if isinstance(cls, type):
        if name.startswith("_"): continue  # skip private class
        env_id = name.replace("_", "-")
        gym.register(
            id=env_id,
            entry_point=f"{__name__}.env:Env",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{task_env_cfg.__name__}:{name}"},  
        )

# ## AMP
# gym.register(
#     id="AMP-InterHuman-2Robots",
#     entry_point=f"{__name__}.env:Env",
#     disable_env_checker=True,
#     kwargs={"env_cfg_entry_point": f"{__name__}.task_env_cfg:AMP_InterHuman_2Robots"},
# )
# gym.register(
#     id="AMP-InterHuman",
#     entry_point=f"{__name__}.env:Env",
#     disable_env_checker=True,
#     kwargs={"env_cfg_entry_point": f"{__name__}.task_env_cfg:AMP_InterHuman"},
# )
# gym.register(
#     id="AMP-Humanoid",
#     entry_point=f"{__name__}.env:Env",
#     disable_env_checker=True,
#     kwargs={"env_cfg_entry_point": f"{__name__}.task_env_cfg:AMP_Humanoid",
# },
# )

# ## AIP
# gym.register(
#     id="AIP-InterHuman-2Robots",
#     entry_point=f"{__name__}.env:Env",
#     disable_env_checker=True,
#     kwargs={"env_cfg_entry_point": f"{__name__}.task_env_cfg:AIP_InterHuman_2Robots"},
# )

# ### PPO
# gym.register(
#     id="PPO-InterHuman",
#     entry_point=f"{__name__}.env:Env",
#     disable_env_checker=True,
#     kwargs={"env_cfg_entry_point": f"{__name__}.task_env_cfg:PPO_InterHuman"},
# )

# gym.register(
#     id="PPO-Humanoid",
#     entry_point=f"{__name__}.env:Env",
#     disable_env_checker=True,
#     kwargs={"env_cfg_entry_point": f"{__name__}.task_env_cfg:PPO_Humanoid"},
# )

# gym.register(
#     id="PPO-InterHuman-2Robots",
#     entry_point=f"{__name__}.env:Env",
#     disable_env_checker=True,
#     kwargs={"env_cfg_entry_point": f"{__name__}.task_env_cfg:PPO_InterHuman_2Robots"},
# )


# ### HRL
# gym.register(
#     id="HRL-InterHuman",
#     entry_point=f"{__name__}.env:Env",
#     disable_env_checker=True,
#     kwargs={"env_cfg_entry_point": f"{__name__}.task_env_cfg:HRL_InterHuman"},
# )
# gym.register(
#     id="HRL-Humanoid",
#     entry_point=f"{__name__}.env:Env",
#     disable_env_checker=True,
#     kwargs={"env_cfg_entry_point": f"{__name__}.task_env_cfg:HRL_Humanoid"},
# )