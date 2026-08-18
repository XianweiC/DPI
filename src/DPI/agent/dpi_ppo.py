import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import distributions as D

from DPI.model.preference import PreferenceEncoder
from DPI.model.actor_critic import ConditionalActorCritic, ContinuousActorCritic
from DPI.agent.base import BaseAgent


class DPI_PPO(BaseAgent):
    def __init__(
        self, 
        observation_sapce, 
        action_dim,
        hidden_szie, 
        reward_size, 
        network_type,
        action_type,
        learning_rate=2.5e-4, 
        beta=0.5, 
        gamma=0.5,
        dir=0.5,
        kl=0.1,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__(
            observation_sapce, 
            action_dim, 
            hidden_szie, 
            reward_size, 
            device
        )

        self.beta = beta 
        self.gamma = gamma
        self.alpha_dir = dir
        self.alpha_kl = kl

        self.MAX_GRAD_NORM = 0.5

        self.observation_sapce = observation_sapce
        self.action_dim = action_dim
        self.hidden_szie = hidden_szie
        self.reward_size = reward_size
        self.learning_rate = learning_rate

        self.preference_inference_module = PreferenceEncoder(
            observation_sapce, 
            hidden_szie, 
            reward_size, 
            network_type).to(self.device)

        if action_type == 'discrete':
            self.action_module = ConditionalActorCritic(
                observation_sapce, 
                action_dim, 
                hidden_szie, 
                reward_size, 
                network_type).to(self.device)
        elif action_type == 'continuous':
            self.action_module = ContinuousActorCritic(
                observation_sapce, 
                action_dim, 
                hidden_szie, 
                reward_size, 
                network_type).to(self.device)
        else:
            raise ValueError(f"Invalid action type: {action_type}")

        self.preference_optimizer = optim.Adam(self.preference_inference_module.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.action_module.parameters(), lr=learning_rate)

    @torch.no_grad()
    def act(self, history, K: int = 8):
        history, obs = self._prepare_history(history)
        B = history.shape[0]

        omega_one, mu, logstd = self.preference_inference_module(history)
        std = torch.exp(logstd)
        eps = torch.randn(B, K, mu.size(-1), device=self.device)
        omegas = mu.unsqueeze(1) + eps * std.unsqueeze(1)
        omegas = F.softmax(omegas, dim=-1)

        scores = []
        for k in range(K):
            omega_k = omegas[:, k, :]
            _, _, _, vvec_k = self.action_module(obs, omega_k) 
            score_k = (vvec_k * omega_k).sum(dim=-1) 
            scores.append(score_k)
        scores = torch.stack(scores, dim=1)
        idx = scores.argmax(dim=1)
        omega_star = omegas[torch.arange(B, device=self.device), idx]  

        action, log_prob, entropy, _ = self.action_module(obs, omega_star)

        return action, log_prob, entropy, omega_star, mu, logstd, idx

    def learn(
        self, 
        history, 
        actions, 
        old_log_probs, 
        vec_returns, 
        advantages, 
        preferences, 
        clip_eps=0.2, 
        vf_coef=0.5, 
        ent_coef=0.01,
        coef_lambda = 0.1, 
        coef_gamma = 0.01
    ):
        history, obs = self._prepare_history(history)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        vec_returns = vec_returns.to(self.device)
        advantages = advantages.to(self.device)
        preferences = preferences.to(self.device)

        self.preference_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        log_probs, entropy, vec_values = self.action_module.evaluate_actions(obs, preferences, actions)  

        scalar_values = (vec_values * preferences).sum(dim=-1)      
        scalar_returns = (vec_returns * preferences).sum(dim=-1)    

        ratio = torch.exp(log_probs - old_log_probs)                
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        actor_loss = - torch.min(surr1, surr2).mean()

        critic_loss_l1 = F.mse_loss(vec_values, vec_returns)
        critic_loss_l2 = F.mse_loss(scalar_values, scalar_returns)
        critic_loss = self.beta * critic_loss_l1 + (1 - self.beta) * critic_loss_l2

        policy_loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy.mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.action_module.parameters(), self.MAX_GRAD_NORM)
        self.policy_optimizer.step()

        omega_pred, mu_pred, logstd_pred = self.preference_inference_module(history)
        std = torch.exp(logstd_pred)
        omega_dist = D.Normal(mu_pred, std)
        prior = D.Normal(loc=torch.zeros_like(mu_pred), scale=torch.ones_like(std))
        kl_loss = D.kl_divergence(omega_dist, prior).sum(-1).mean()

        scalarized_ret = (vec_returns * omega_pred).sum(-1).mean()
        elbo = -(scalarized_ret - self.alpha_kl * kl_loss)

        vec_norm = vec_returns.norm(p=2, dim=-1, keepdim=True)
        valid_mask = (vec_norm.squeeze(-1) > 1e-8).float()
        td_dir = vec_returns / (vec_norm + 1e-8)
        omega_dir = F.normalize(omega_pred, dim=-1, eps=1e-8)
        cos_sim = F.cosine_similarity(omega_dir, td_dir, dim=-1, eps=1e-8)
        direction_loss = ((1.0 - cos_sim) * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        stabilize_loss = F.mse_loss(omega_pred, preferences)

        prefer_loss = elbo + coef_lambda * direction_loss + coef_gamma * stabilize_loss

        self.preference_optimizer.zero_grad()
        prefer_loss.backward()
        nn.utils.clip_grad_norm_(self.preference_inference_module.parameters(), self.MAX_GRAD_NORM)
        self.preference_optimizer.step()

        return actor_loss, critic_loss_l1, critic_loss_l2, kl_loss, direction_loss, stabilize_loss
