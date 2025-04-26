from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch
from torch import nn

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, params, CLIP=None, device=None):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, 
                              clip_actions=False,
                              clip_log_std=True,
                              min_log_std=-20.0,
                              max_log_std=2.0,
                              )

        self.net = nn.Sequential(
            nn.Linear(observation_space, params),
            nn.ReLU(),
            nn.Linear(params, int(params/2)),
            nn.ReLU(),
            nn.Linear(int(params/2), action_space)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space)) 

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

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

class Discriminator(DeterministicMixin, Model):
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
    
def instantiate_AMP(env, params: int=1, device: torch.device | None=None):
    models = {}
    models["policy"] = Policy(env.observation_space.shape[0], env.action_space.shape[0], params=params, device=device)
    models["value"] = Value(env.observation_space.shape[0], env.action_space.shape[0], params=params, device=device)
    models["discriminator"] = Discriminator(env.amp_observation_size, env.action_space.shape[0], params=params, device=device)
    return models

def instantiate_AMP_2robots(env, params: int=1024, device: torch.device | None=None):
    models = {}
    models["policy1"] = Policy(int(env.observation_space.shape[0]/2), int(env.action_space.shape[0]/2), params=params, device=device)
    models["value1"] = Value(int(env.observation_space.shape[0]/2), int(env.action_space.shape[0]/2), params=params, device=device)
    models["policy2"] = Policy(int(env.observation_space.shape[0]/2), int(env.action_space.shape[0]/2), params=params, device=device)
    models["value2"] = Value(int(env.observation_space.shape[0]/2), int(env.action_space.shape[0]/2), params=params, device=device)
    models["discriminator"] = Discriminator(env.amp_observation_size, env.action_space.shape[0], params=params, device=device)
    return models

def get_AMP_policy_model():
    return Policy

def instantiate_AMP_policy(env, params: int=1, device: torch.device | None=None):
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0], params=params, device=device)
    return policy

def instantiate_AMP_discriminator(env, params: int=1, device: torch.device | None=None):
    policy = Discriminator(env.amp_observation_size, env.action_space.shape[0], params=params, device=device)
    return policy