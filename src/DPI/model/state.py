from torch import nn

from DPI.model.network import build_mlp, build_cnn


class StateEncoder1D(nn.Module):
    def __init__(self, obs_dim, hidden_size, network_type='queue'):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_size, hidden_size, network_type)

    def forward(self, x):
        return self.net(x)


class StateEncoder2D(nn.Module):
    def __init__(self, obs_dim, hidden_size, network_type='maze'):
        super().__init__()
        self.net = build_cnn(obs_dim, hidden_size, network_type)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    from gymnasium import spaces
    import torch
    
    batch_size = 128
    win_size = 8
    hidden_size = 128
    
    # Queue
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(win_size, 6))
    history = torch.randn(batch_size, win_size, 6).cuda()
    model = StateEncoder1D(obs_space.shape[1], hidden_size, network_type='queue').cuda()
    x = model(history)
    print(x.shape)
