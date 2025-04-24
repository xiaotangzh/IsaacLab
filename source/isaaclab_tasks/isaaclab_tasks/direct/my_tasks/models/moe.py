from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
from torch import nn

# ==================== Policy Model (Gaussian Policy) ====================
class Policy(GaussianMixin, Model):
    def __init__(self, states_goal_size, gating_size, CLIP=None, device=None):
        Model.__init__(self, states_goal_size, gating_size, device)
        GaussianMixin.__init__(self, 
                              clip_actions=False,
                              clip_log_std=True,
                              min_log_std=-20.0,
                              max_log_std=2.0,
                              )

        # Network layers (1024 -> 512)
        self.net = nn.Sequential(
            nn.Linear(states_goal_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, gating_size) 
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(gating_size)) 

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["goal"]], dim=-1)), self.log_std_parameter, {}

# ==================== Value Model (Deterministic) ====================
class Value(DeterministicMixin, Model):
    def __init__(self, states_goal_size, gatings_size, device=None):
        Model.__init__(self, states_goal_size, gatings_size, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # Network layers (1024 -> 512 -> 1)
        self.net = nn.Sequential(
            nn.Linear(states_goal_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["goal"]], dim=-1)), {}  

# ==================== Discriminator Model (Deterministic) ====================
class Expert(DeterministicMixin, Model):
    def __init__(self, gating_size, sub_action_size, device=None):
        Model.__init__(self, gating_size, sub_action_size, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        # Network layers (same as Value network)
        self.net = nn.Sequential(
            nn.Linear(gating_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, sub_action_size) 
        )

    def compute(self, inputs, role):
        return self.net(inputs["gating"]), {}  
    
def instantiate_MOE(env, params, cfg, device):
    models = {}
    models["policy"] = Policy(env.observation_space.shape[0], env.action_space.shape[0], device)
    models["value"] = Value(env.observation_space.shape[0], env.action_space.shape[0], device)
    models["experts"] = [Expert(gating_size=cfg["gating_size"], sub_action_space=sub_action_size, device=device) for expert, sub_action_size in cfg["sub_action_size"]]

    return models