import numpy as np

import torch

from DPI.misc.utils import AverageMeter


def compute_gae_vec(reward_vec, done, value_vec, next_value_vec, gamma=0.99, gae_lambda=0.95, clip_adv=None):
    reward_vec = np.asarray(reward_vec, np.float32)
    done = np.asarray(done, np.float32)
    value_vec = np.asarray(value_vec, np.float32)
    next_value_vec = np.asarray(next_value_vec, np.float32)

    T, E, d = reward_vec.shape
    vec_adv = np.zeros((T, E, d), dtype=np.float32)
    gae = np.zeros((E, d), dtype=np.float32)

    for t in range(T - 1, -1, -1):
        not_done = (1.0 - done[t])[:, None]
        delta = reward_vec[t] + gamma * next_value_vec[t] * not_done - value_vec[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        vec_adv[t] = gae

    if clip_adv is not None:
        np.clip(vec_adv, -clip_adv, clip_adv, out=vec_adv)

    vec_ret = vec_adv + value_vec
    return vec_adv.astype(np.float32), vec_ret.astype(np.float32)


def scalarize_adv_onpolicy(vec_adv, omega_star, standardize=True):
    scalar_adv = np.sum(vec_adv * omega_star, axis=-1)  
    if standardize:
        mean = scalar_adv.mean()
        std = scalar_adv.std()
        scalar_adv = (scalar_adv - mean) / (std + 1e-8)
    return scalar_adv.astype(np.float32)


def train(agent, buf, ppo_epochs=4, mini_bs=4096, gamma=0.99, lam=0.95, device="cuda"):
    actor_meter = AverageMeter()
    critic1_meter = AverageMeter()
    critic2_meter = AverageMeter()
    kl_meter = AverageMeter()
    scalar_meter = AverageMeter()
    direction_meter = AverageMeter()

    T, E = buf.T, buf.E
    d = buf.reward_vec.shape[-1]

    vec_adv, vec_ret = compute_gae_vec(
        reward_vec=buf.reward_vec,
        done=buf.done,
        value_vec=buf.value_vec,
        next_value_vec=buf.next_value_vec,
        gamma=gamma, gae_lambda=lam, clip_adv=10.0
    )
    scalar_adv = scalarize_adv_onpolicy(vec_adv, buf.omega_star, standardize=True)  

    def flat(x): return x.reshape(T*E, *x.shape[2:]) if x.ndim > 2 else x.reshape(T*E)
    obs = flat(buf.obs)
    actions = flat(buf.actions)
    old_logp = flat(buf.log_probs)
    pref = flat(buf.omega_star)
    vec_returns = flat(vec_ret)
    adv = flat(scalar_adv)

    obs_t = torch.from_numpy(obs).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    old_logp_t = torch.from_numpy(old_logp).to(device)
    pref_t = torch.from_numpy(pref).to(device)
    vec_returns_t = torch.from_numpy(vec_returns).to(device)
    adv_t = torch.from_numpy(adv).to(device)

    N = obs_t.shape[0]
    idx_all = np.arange(N)

    for _ in range(ppo_epochs):
        np.random.shuffle(idx_all)
        for start in range(0, N, mini_bs):
            idx = idx_all[start:start+mini_bs]
            batch_hist = obs_t[idx]
            batch_act  = actions_t[idx]
            batch_old  = old_logp_t[idx]
            batch_pref = pref_t[idx]
            batch_vret = vec_returns_t[idx]
            batch_adv  = adv_t[idx]

            loss_stats = agent.learn(
                history=batch_hist,
                actions=batch_act,
                old_log_probs=batch_old,
                vec_returns=batch_vret,
                advantages=batch_adv,
                preferences=batch_pref,
                clip_eps=0.2, 
                vf_coef=0.5, 
                ent_coef=0.01)

            actor_loss, critic_loss_l1, critic_loss_l2, kl_loss, direction_loss, stabilize_loss = loss_stats

            actor_meter.update(actor_loss.item(), mini_bs)
            critic1_meter.update(critic_loss_l1.item(), mini_bs)
            critic2_meter.update(critic_loss_l2.item(), mini_bs)
            kl_meter.update(kl_loss.item(), mini_bs)
            direction_meter.update(direction_loss.item(), mini_bs)
            scalar_meter.update(stabilize_loss.item(), mini_bs)

    return actor_meter, critic1_meter, critic2_meter, kl_meter, direction_meter, scalar_meter
