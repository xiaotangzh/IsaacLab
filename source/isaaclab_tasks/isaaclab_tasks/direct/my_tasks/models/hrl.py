from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
from torch import nn

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, params, device=None):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, 
                              clip_actions=False,
                              clip_log_std=True,
                              min_log_std=-20.0,
                              max_log_std=2.0,
                              )

        self.net = nn.Sequential(
            nn.Linear(observation_space+action_space, params),
            nn.ReLU(),
            nn.Linear(params, int(params/2)),
            nn.ReLU(),
            nn.Linear(int(params/2), action_space) 
        )
        # self.log_std_parameter = nn.Parameter(torch.zeros(action_space)) 

        # zero initialize
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_std_parameter = nn.Parameter(torch.full((action_space,), -3.0))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class JustPretrainedPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, params, pretrained_policy: Model, device=None):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, 
                              clip_actions=False,
                              clip_log_std=True,
                              min_log_std=-20.0,
                              max_log_std=2.0,
                              )
        self.policy = pretrained_policy
        self.policy.eval()

    def compute(self, inputs, role):
        with torch.no_grad():
            return self.policy(inputs)

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, params, device=None):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(
            nn.Linear(observation_space, params),
            nn.ReLU(),
            nn.Linear(params, int(params/2)),
            nn.ReLU(),
            nn.Linear(int(params/2), 1) 
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        return self.net(states), {} 
    
def instantiate_HRL(env, params: int=1, device: torch.device | None=None):
    models = {}
    models["policy"] = Policy(env.observation_space.shape[0], env.action_space.shape[0], params=params, device=device)
    models["value"] = Value(env.observation_space.shape[0], env.action_space.shape[0], params=params, device=device)
    return models