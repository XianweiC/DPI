import torch
from torch import nn
from torch.nn import functional as F

from DPI.model.state import StateEncoder1D, StateEncoder2D
from DPI.model.network import build_gru


class PreferenceEncoder(nn.Module):
    """Encoder to infer preference vector omega_t from observation history."""
    def __init__(self, obs_space, hidden_size, reward_size, network_type):
        super().__init__()

        obs_channel = obs_space.shape[1]
        len_obs = len(obs_space.shape)

        if len_obs == 2:
            self.feature = StateEncoder1D(obs_channel, hidden_size, network_type)
        elif len_obs == 4:
            self.feature = StateEncoder2D(obs_channel, hidden_size, network_type)
        else:
            raise ValueError(f"Invalid observation shape: {obs_space.shape}")

        self.memo = build_gru(hidden_size, hidden_size)
        
        if len_obs == 2:
            self.linear_mu = nn.Linear(hidden_size, reward_size)
            self.linear_logstd = nn.Linear(hidden_size, reward_size)
        elif len_obs == 4:
            self.linear_mu = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, reward_size)
            )
            self.linear_logstd = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, reward_size)
            )

    def forward(self, history):        
        if len(history.shape) == 5:
            
            B, seq_len, C, H, W = history.shape
            obs = history.view(B*seq_len, C, H, W)
        elif len(history.shape) == 3:
            B, seq_len, D = history.shape
            history = history.unsqueeze(0)
            obs = history.view(B*seq_len, D)

        x = self.feature(obs)
        x = x.view(B, seq_len, -1)
        _, ht = self.memo(x)
        x = ht[-1]

        mu = self.linear_mu(x)   
        logstd = self.linear_logstd(x)  
        logstd = torch.clamp(logstd, min=-4, max=2)  

        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        z = mu + eps * std

        omega = F.softmax(z, dim=-1)  
        return omega, mu, logstd


if __name__ == '__main__':
    from gymnasium import spaces
    
    batch_size = 128
    win_size = 8
    hidden_size = 128
    
    
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(win_size, 6))
    action_space = spaces.Discrete(2)
    reward_size = 5
    history = torch.randn(batch_size, win_size, 6).cuda()
    model = PreferenceEncoder(obs_space, hidden_size, reward_size, network_type='queue').cuda()
    omega, mu, logstd = model(history)
    assert omega.shape == (batch_size, reward_size)
    assert mu.shape == (batch_size, reward_size)
    assert logstd.shape == (batch_size, reward_size)
    print("Queue tests passed!")

    
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(win_size, 6, 84, 84))
    action_space = spaces.Discrete(5)
    reward_size = 5
    history = torch.randn(batch_size, win_size, 6, 84, 84).cuda()
    model = PreferenceEncoder(obs_space, hidden_size, reward_size, network_type='maze').cuda()
    omega, mu, logstd = model(history)
    assert omega.shape == (batch_size, reward_size)
    assert mu.shape == (batch_size, reward_size)
    assert logstd.shape == (batch_size, reward_size)
    print("Maze tests passed!")

    
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(win_size, 17))
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
    reward_size = 3
    history = torch.randn(batch_size, win_size, 17).cuda()
    model = PreferenceEncoder(obs_space, hidden_size, reward_size, network_type='half-cheetah').cuda()
    omega, mu, logstd = model(history)
    assert omega.shape == (batch_size, reward_size)
    assert mu.shape == (batch_size, reward_size)
    assert logstd.shape == (batch_size, reward_size)
    print("Half-Cheetah tests passed!")
