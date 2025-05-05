import torch

class Bridge:
    '''
    Bridge class to connect the environment and agent.
    '''
    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self.timestep: int = 0
        self.timesteps: int = 0
        self.episode_length: torch.Tensor = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.terminates: torch.Tensor | None = None # [num_envs, ] terminate the environment from agent

    def set_timestep(self, timestep, timesteps):
        self.timestep = timestep
        if self.timesteps <= 0: self.timesteps = timesteps 

    def set_episode_length(self, episode_length):
        self.episode_length = episode_length

    def get_episode_length(self):
        return self.episode_length

    def set_terminates(self, terminates):
        self.terminates = terminates

    def get_terminates(self) -> torch.Tensor | None:
        if self.terminates is None: return None

        terminates = self.terminates
        self.terminates = None
        return terminates