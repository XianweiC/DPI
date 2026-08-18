import torch
from torch import nn
from torch import distributions as D

from DPI.model.state import StateEncoder1D, StateEncoder2D


class ConditionalActorCritic(nn.Module):
    def __init__(
        self, 
        observation_space, 
        action_dim, 
        hidden_size, 
        reward_size, 
        network_type
    ):
        super().__init__()
        obs_channel = observation_space.shape[1]
        len_obs = len(observation_space.shape)

        if len_obs == 2:
            self.feature = StateEncoder1D(obs_channel, hidden_size, network_type)
        elif len_obs == 4:
            self.feature = StateEncoder2D(obs_channel, hidden_size, network_type)
        else:
            raise ValueError(f"Invalid observation shape: {observation_space.shape}")

        self.actor = nn.Sequential(
            nn.Linear(hidden_size+reward_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size+reward_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, reward_size),
        )

    def forward(self, x, omega):
        x = self.feature(x)
        x = torch.cat((x, omega), dim=1)
        logits = self.actor(x)
        dist = D.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(x)
        return action, log_prob, entropy, value

    def evaluate_actions(self, x, omega, actions):
        x = self.feature(x)
        x = torch.cat((x, omega), dim=1)
        logits = self.actor(x)
        dist = D.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.critic(x).squeeze(-1)

        return log_probs, entropy, value


class ContinuousActorCritic(ConditionalActorCritic):
    EPS = 1e-6

    def __init__(
        self, 
        observation_space, 
        action_dim, 
        hidden_size, 
        reward_size, 
        network_type
    ):
        super().__init__(
            observation_space, 
            action_dim, 
            hidden_size, 
            reward_size, 
            network_type
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _distribution(self, x, omega):
        x = self.feature(x)
        x = torch.cat((x, omega), dim=1)

        mu = self.actor(x)
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mu)
        dist = D.Normal(mu, log_std.exp())
        value = self.critic(x)

        return dist, value

    def _log_prob(self, dist, raw_action, action):
        
        return (
            dist.log_prob(raw_action)
            - torch.log(1.0 - action.pow(2) + self.EPS)
        ).sum(dim=-1)

    def forward(self, x, omega):
        dist, value = self._distribution(x, omega)

        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._log_prob(dist, raw_action, action)

        
        entropy = -log_prob

        return action, log_prob, entropy, value

    def evaluate_actions(self, x, omega, actions):
        dist, value = self._distribution(x, omega)

        
        actions = actions.clamp(
            -1.0 + self.EPS,
            1.0 - self.EPS,
        )
        raw_actions = torch.atanh(actions)
        log_probs = self._log_prob(dist, raw_actions, actions)

        
        entropy_raw = dist.rsample()
        entropy_action = torch.tanh(entropy_raw)
        entropy = -self._log_prob(
            dist,
            entropy_raw,
            entropy_action,
        )

        return log_probs, entropy, value


if __name__ == '__main__':
    from gymnasium import spaces
    
    batch_size = 128
    win_size = 8
    hidden_size = 128
    
    
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(win_size, 6))
    action_space = spaces.Discrete(2)
    reward_size = 5
    model = ConditionalActorCritic(obs_space, action_space.n, hidden_size, reward_size, network_type='queue').cuda()
    obs = torch.randn(batch_size, 6).cuda()
    omega = torch.randn(batch_size, reward_size).cuda()
    action, log_prob, entropy, value = model(obs, omega)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, reward_size)
    print("Queue tests passed!")

    
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(win_size, 6, 84, 84))
    action_space = spaces.Discrete(5)
    reward_size = 5
    model = ConditionalActorCritic(obs_space, action_space.n, hidden_size, reward_size, network_type='maze').cuda()
    obs = torch.randn(batch_size, 6, 84, 84).cuda()
    omega = torch.randn(batch_size, reward_size).cuda()
    action, log_prob, entropy, value = model(obs, omega)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, reward_size)
    print("Maze tests passed!")

    
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(win_size, 17))
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
    reward_size = 3
    model = ConditionalActorCritic(obs_space, action_space.shape[0], hidden_size, reward_size, network_type='half-cheetah').cuda()
    obs = torch.randn(batch_size, 17).cuda()
    omega = torch.randn(batch_size, reward_size).cuda()
    action, log_prob, entropy, value = model(obs, omega)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size, reward_size)
    print("Half-Cheetah tests passed!")
