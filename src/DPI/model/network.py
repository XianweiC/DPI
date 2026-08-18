from torch import nn


def build_mlp(input_size, hidden_size, output_size, network_type='queue'):
    if network_type == 'queue':
        net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    elif network_type == 'half-cheetah':
        net = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), 
            nn.Tanh(),
        )
    else:
        raise ValueError(f"Invalid network type: {network_type}")
    return net


def build_cnn(obs_dim, hidden_size, network_type='maze'):
    if network_type == 'unknown':
        net = nn.Sequential(
            nn.Conv2d(obs_dim, 16, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(5 * 5 * 64, hidden_size), 
            nn.LeakyReLU(),
        )
    elif network_type == 'maze':
        # net = nn.Sequential(
        #     nn.Conv2d(in_channels=obs_dim, out_channels=32, kernel_size=8, stride=4),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        #     nn.LeakyReLU(),
        #     nn.Flatten(1),
        #     nn.Linear(7 * 7 * 64, hidden_size), 
        #     nn.LeakyReLU(),
        # )

        net = nn.Sequential(
            nn.Conv2d(obs_dim, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, hidden_size),
            nn.LeakyReLU(),
        )

    else: 
        raise ValueError(f"Invalid network type: {network_type}")
    return net


def build_gru(obs_dim, hidden_size):
    return nn.GRU(obs_dim, hidden_size, batch_first=True)


def build_mlp_net(hidden_size, output_size):
    return nn.Linear(hidden_size, output_size)
