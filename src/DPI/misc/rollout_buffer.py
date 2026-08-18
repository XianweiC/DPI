import numpy as np
import torch

from DPI.env.maze_env import build_layout


class RolloutBuffer:
    def __init__(self, T, E, obs_shape, action_space, d):
        
        self.T, self.E = T, E
        self.obs = np.zeros((T, E) + obs_shape, dtype=np.float32)   
        if action_space != 8 and action_space != 6:
            self.actions = np.zeros((T, E), dtype=np.int64)
        else: 
            self.actions = np.zeros((T, E, action_space), dtype=np.float32)
        self.log_probs = np.zeros((T, E), dtype=np.float32)
        self.done = np.zeros((T, E), dtype=np.float32)
        self.reward_vec = np.zeros((T, E, d), dtype=np.float32)
        self.omega_star = np.zeros((T, E, d), dtype=np.float32)     
        self.mu = np.zeros((T, E, d), dtype=np.float32)
        self.logstd = np.zeros((T, E, d), dtype=np.float32)

        
        self.value_vec = np.zeros((T, E, d), dtype=np.float32)
        self.next_value_vec = np.zeros((T, E, d), dtype=np.float32)

        self.ptr = 0

    def store(self, t, obs, action, logp, done, rvec, omegastar, mu, logstd, vvec, vvec_next):
        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = logp
        self.done[t] = done
        self.reward_vec[t] = rvec
        self.omega_star[t] = omegastar
        self.mu[t] = mu
        self.logstd[t] = logstd
        self.value_vec[t] = vvec
        self.next_value_vec[t] = vvec_next


def get_value_vec(agent, obs_curr, omega):
    
    with torch.no_grad():
        x = agent.action_module.feature(obs_curr)
        x = torch.cat((x, omega), dim=1)
        vvec = agent.action_module.critic(x)  
    return vvec


def sample_queue_schedule(rng: np.random.RandomState):
    t_a = int(rng.randint(2, 4))  
    t_b = int(rng.randint(4, 6))  
    t_c = int(rng.randint(6, 8))  

    return [
        ("morality_surge", t_a, {"mult": 2.5}),
        ("energy_drought", t_b, {"mult": 2.0}),
        ("deadline_shock", t_c, {"shrink_ratio": 0.6}),
    ]


def sample_maze_schedule(rng: np.random.RandomState):
    t_a = int(rng.randint(30, 50))
    t_b = int(rng.randint(70, 90))
    t_c = int(rng.randint(105, 115))

    return [
        (t_a, "hazard_surge"),
        (t_b, "energy_drought"),
        (t_c, "deadline_shock"),
    ]


def sample_halfcheetah_schedule(rng: np.random.RandomState):
    t_a = int(rng.randint(40, 60))
    t_b = int(rng.randint(90, 110))
    t_c = int(rng.randint(140, 160))

    return [
        (t_a, "speed_surge"),
        (t_b, "energy_drought"),
        (t_c, "deadline_shock"),
    ]

def queue_rollout(env, agent, buf, T, K, device):
    obs, info = env.reset(options={'schedule': sample_queue_schedule(env.rng)})

    for t in range(T):
        obs_tensor = torch.from_numpy(obs).to(device)            

        action, logp, ent, omega_star, mu, logstd, idx = agent.act(obs_tensor, K=K)

        action_np = action.detach().cpu().numpy()
        logp_np = logp.detach().cpu().numpy()
        omega_star_np = omega_star.detach().cpu().numpy()
        mu_np = mu.detach().cpu().numpy()
        logstd_np = logstd.detach().cpu().numpy()

        
        if len(obs_tensor.shape) == 2:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_curr_single = obs_tensor[:, -1].to(device)                 
        vvec = get_value_vec(agent, obs_curr_single, omega_star).cpu().numpy()  

        next_obs_batch, rewards, terminated, truncated, info = env.step(action_np)

        done = np.logical_or(terminated, truncated).astype(np.float32)
        reward_vec = info['mor']

        
        next_obs_tensor = torch.from_numpy(next_obs_batch).to(device)
        if len(next_obs_tensor.shape) == 2:
            next_obs_tensor = next_obs_tensor.unsqueeze(0)
        next_obs_curr = next_obs_tensor[:, -1]
        vvec_next = get_value_vec(agent, next_obs_curr, omega_star).cpu().numpy()  

        buf.store(t, obs, action_np, logp_np, done, reward_vec, omega_star_np, mu_np, logstd_np, vvec, vvec_next)

        obs = next_obs_batch

        
        if done:
            
            obs, info = env.reset(options={'schedule': sample_queue_schedule(env.rng)})

    return buf


def maze_rollout(env, agent, buf, T, K, device):
    layout = build_layout(gh=11, gw=11, corridor_width=1, hazard_band_width=1, margin=1)
    obs, info = env.reset(options={'layout': layout, 'schedule': sample_maze_schedule(env.rng)})

    
    for t in range(T):
        obs_tensor = torch.from_numpy(obs).to(device)            

        action, logp, ent, omega_star, mu, logstd, idx = agent.act(obs_tensor, K=K)

        action_np = action.detach().cpu().numpy()
        logp_np = logp.detach().cpu().numpy()
        omega_star_np = omega_star.detach().cpu().numpy()
        mu_np = mu.detach().cpu().numpy()
        logstd_np = logstd.detach().cpu().numpy()

        
        if len(obs_tensor.shape) == 4:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_curr_single = obs_tensor[:, -1].to(device)                 
        vvec = get_value_vec(agent, obs_curr_single, omega_star).cpu().numpy()  

        next_obs_batch, rewards, terminated, truncated, info = env.step(action_np)

        done = np.logical_or(terminated, truncated).astype(np.float32)
        reward_vec = info['mor']

        
        next_obs_tensor = torch.from_numpy(next_obs_batch).to(device)
        if len(next_obs_tensor.shape) == 4:
            next_obs_tensor = next_obs_tensor.unsqueeze(0)
        next_obs_curr = next_obs_tensor[:, -1]
        vvec_next = get_value_vec(agent, next_obs_curr, omega_star).cpu().numpy()  

        buf.store(t, obs, action_np, logp_np, done, reward_vec, omega_star_np, mu_np, logstd_np, vvec, vvec_next)

        obs = next_obs_batch

        
        if done:
            obs, info = env.reset(options={'layout': layout, 'schedule': sample_maze_schedule(env.rng)})

    return buf


def halfcheetah_rollout(env, agent, buf, T, K, device):
    obs, info = env.reset(options={'schedule': sample_halfcheetah_schedule(env.rng)})

    for t in range(T):
        
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        

        action, logp, ent, omega_star, mu, logstd, idx = agent.act(obs_tensor, K=K)

        action_np = action.detach().cpu().numpy()
        logp_np = logp.detach().cpu().numpy()
        omega_star_np = omega_star.detach().cpu().numpy()
        mu_np = mu.detach().cpu().numpy()
        logstd_np = logstd.detach().cpu().numpy()

        
        if len(obs_tensor.shape) == 2:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_curr_single = obs_tensor[:, -1].to(device)                 
        vvec = get_value_vec(agent, obs_curr_single, omega_star).cpu().numpy()  

        next_obs_batch, rewards, terminated, truncated, info = env.step(action_np[0])
        done = np.logical_or(terminated, truncated).astype(np.float32)
        reward_vec = info['mor']

        
        next_obs_tensor = torch.from_numpy(next_obs_batch).to(device)
        if len(next_obs_tensor.shape) == 2:
            next_obs_tensor = next_obs_tensor.unsqueeze(0)
        next_obs_curr = next_obs_tensor[:, -1]
        vvec_next = get_value_vec(agent, next_obs_curr, omega_star).cpu().numpy()  

        buf.store(t, obs_np[np.newaxis, ...], action_np, logp_np, done, reward_vec, omega_star_np, mu_np, logstd_np, vvec, vvec_next)

        obs = next_obs_batch

        
        if done:
            obs, info = env.reset(options={'schedule': sample_halfcheetah_schedule(env.rng)})

    return buf
