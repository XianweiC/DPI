import torch


class BaseAgent:
    def __init__(
        self, 
        observation_space, 
        action_space,
        hidden_size, 
        reward_size,
        device
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.reward_size = reward_size
        self.device = device

    def _prepare_history(self, history):
        if not torch.is_tensor(history):
            history = torch.as_tensor(history, dtype=torch.float32)
        history = history.to(self.device)
        
        if len(history.shape) == 4 or len(history.shape) == 2:
            history = history.unsqueeze(0)
        obs = history[:, -1]

        return history, obs

    def act(self, obs, K=None):
        pass

    def learn(self, obs, action, reward, next_obs):
        pass
