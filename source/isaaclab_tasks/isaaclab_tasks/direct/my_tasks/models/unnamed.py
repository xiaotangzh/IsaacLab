from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
from torch import nn

# ==================== Policy Model (Gaussian Policy) ====================
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, params, CLIP=None, device=None):
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
            nn.Linear(observation_space, params),
            nn.ReLU(),
            nn.Linear(params, int(params/2)),
            nn.ReLU(),
            nn.Linear(int(params/2), action_space)  # Output actions
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space)) #todo different from default yaml

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["latents"]], dim=-1)), self.log_std_parameter, {}

# ==================== Value Model (Deterministic) ====================
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, params, device=None):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # Network layers (1024 -> 512 -> 1)
        self.net = nn.Sequential(
            nn.Linear(observation_space, params),
            nn.ReLU(),
            nn.Linear(params, int(params/2)),
            nn.ReLU(),
            nn.Linear(int(params/2), 1)  # Output single value
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["latents"]], dim=-1)), {}  # (value, None)
    
def instantiate_PPO(env, params: int=1, device: torch.device | None=None):
    models = {}
    models["policy"] = Policy(env.observation_space.shape[0], env.action_space.shape[0], params=params, device=device)
    models["value"] = Value(env.observation_space.shape[0], env.action_space.shape[0], params=params, device=device)
    return models