import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional
from collections import deque

import mo_gymnasium as mo_gym


class HalfCheetahEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "HalfCheetahEnv-v0"}
    
    def __init__(self,
                 env_name: str = "mo-halfcheetah-v5",
                 history_size: int = 8,
                 max_steps: int = 200,
                 base_deadline: int = 200,
                 speed_scale: float = 1.0,
                 control_cost_scale: float = 0.1,
                 energy_init: float = 100.0,
                 energy_cost: float = 0.01,
                 deadline_bonus: float = 10.0,
                 enable_events: bool = True,
                 event_min_gap: int = 20,
                 event_prob: float = 0.08,
                 deadline_shock_ratio: float = 0.5,
                 speed_mult: float = 1.5,
                 energy_mult: float = 2.0,
                 schedule_only: bool = False,
                 seed: Optional[int] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.base_env = mo_gym.make(env_name, render_mode=render_mode)
        self.render_mode = render_mode
        
        self.H_win = int(history_size)
        self.max_steps = int(max_steps)
        self.base_deadline0 = int(base_deadline)
        self.speed_scale0 = float(speed_scale)
        self.control_cost_scale0 = float(control_cost_scale)
        self.energy_init0 = float(energy_init)
        self.energy_cost0 = float(energy_cost)
        self.deadline_bonus0 = float(deadline_bonus)
        
        self.enable_events = bool(enable_events)
        self.event_min_gap = int(event_min_gap)
        self.event_prob = float(event_prob)
        self.default_deadline_shock_ratio = float(deadline_shock_ratio)
        self.default_speed_mult = float(speed_mult)
        self.default_energy_mult = float(energy_mult)
        self.schedule_only = bool(schedule_only)
        
        self.rng = np.random.RandomState(seed)
        
        base_obs_shape = self.base_env.observation_space.shape
        base_action_shape = self.base_env.action_space.shape
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.H_win, *base_obs_shape),
            dtype=np.float32
        )
        
        self.action_space = self.base_env.action_space
        
        self.d = 3

        self.scales = {
            'speed': 1.0,
            'control_cost': 1.0,
            'energy': 1.0,
        }

        self.default_schedule = None
        self.schedule = []
        self._schedule_idx = 0
        self._last_event_step = -10**9
        self._pending_event = None
        self.phase = 1
        self.event_id = 'none'

        self.buffer = deque(maxlen=self.H_win)
        
        self.reset()
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        if options and 'schedule' in options and options['schedule'] is not None:
            self._schedule_raw = list(options['schedule'])
        elif self.default_schedule is not None:
            self._schedule_raw = list(self.default_schedule)
        else:
            self._schedule_raw = [
                (40, 'deadline_shock'),
                (80, 'speed_surge'),
                (120, 'energy_drought')
            ]

        self._schedule = []
        for item in self._schedule_raw:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                step = int(item[0])
                etype = str(item[1])
                params = dict(item[2]) if len(item) >= 3 and item[2] is not None else {}
                self._schedule.append({"type": etype, "step": step, "params": params})
        self._schedule.sort(key=lambda e: e["step"])
        self._schedule_idx = 0

        base_obs, base_info = self.base_env.reset(seed=seed)

        self.steps = 0
        self.deadline = int(self.base_deadline0)
        self.energy = float(self.energy_init0)
        self.done = False
        self.reached = False

        self.scales['speed'] = 1.0
        self.scales['control_cost'] = 1.0
        self.scales['energy'] = 1.0

        self._last_event_step = -10**9
        self._pending_event = None
        self.phase = 1
        self.event_id = 'none'

        self.speed_scale = self.speed_scale0
        self.control_cost_scale = self.control_cost_scale0
        self.energy_cost = self.energy_cost0

        self.buffer.clear()
        for _ in range(self.H_win):
            self.buffer.append(base_obs.copy().astype(np.float32))

        self.info = {
            'energy': float(self.energy),
            'deadline': int(self.deadline),
            'mor': np.zeros(self.d, dtype=np.float32),
            'reward_names': ['forward_speed', 'control_cost', 'energy_penalty'],
            'event': None,
            'event_id': self.event_id,
            'phase': self.phase,
            'term_cause': None,
            'success': 0
        }
        
        obs = self._stack_history()
        return obs, self.info
    
    def get_evt_time(self):
        res = []
        for item in self._schedule:
            res.append(item['step'])
        return res
    
    def step(self, action):
        if self.done:
            empty = np.zeros(self.d, dtype=np.float32)
            return self._stack_history(), 0.0, True, False, self.info
        
        self.steps += 1
        self.info.pop("event", None)
        self._pending_event = None

        base_obs, base_reward, base_terminated, base_truncated, base_info = self.base_env.step(action)

        if isinstance(base_reward, (list, np.ndarray)):
            if len(base_reward) >= 2:
                forward_speed = float(base_reward[0])
                control_cost = float(base_reward[1])
            elif len(base_reward) == 1:
                forward_speed = float(base_reward[0])
                control_cost = 0.0
            else:
                forward_speed = 0.0
                control_cost = 0.0
        else:
            forward_speed = float(base_reward)
            control_cost = 0.0

        self._maybe_trigger_scheduled_event()
        if self.enable_events and not self.schedule_only:
            self._maybe_trigger_random_event()

        action_magnitude = float(np.linalg.norm(action))
        energy_consumed = self.energy_cost0 * action_magnitude * self.scales['energy']
        self.energy -= energy_consumed

        self.deadline -= 1

        speed_reward = self.speed_scale0 * self.scales['speed'] * forward_speed
        control_penalty = -self.control_cost_scale0 * self.scales['control_cost'] * abs(control_cost)
        energy_penalty = -energy_consumed

        deadline_obj = 0.0

        timeout = self.deadline <= 0
        exhausted = self.energy <= 0.0
        terminated = False
        truncated = False
        term_cause = None
        
        if base_terminated or timeout or exhausted:
            terminated = True
            if not timeout and not exhausted:
                deadline_obj = self.deadline_bonus0
                term_cause = "goal"
            elif timeout:
                deadline_obj = -self.deadline_bonus0
                term_cause = "timeout"
            elif exhausted:
                energy_penalty += -2.0
                term_cause = "exhausted"
        
        if self.steps >= self.max_steps and not terminated:
            truncated = True
            term_cause = "truncated"

        success_flag = int(not timeout and not exhausted and not base_terminated)

        mor = np.array([
            speed_reward,
            control_penalty,
            energy_penalty + deadline_obj
        ], dtype=np.float32)
        
        scalar_r = float(np.clip(mor.sum(), -50.0, 50.0))

        self.info.update({
            'energy': float(self.energy),
            'deadline': max(0, int(self.deadline)),
            'mor': mor,
            'reward_names': ['forward_speed', 'control_cost', 'energy_penalty'],
            'event': self._pending_event,
            'event_id': self.event_id,
            'phase': self.phase,
            'term_cause': term_cause,
            'success': success_flag
        })

        self.buffer.append(base_obs.copy().astype(np.float32))
        obs = self._stack_history()
        
        self.done = terminated or truncated
        
        return obs, scalar_r, terminated, truncated, self.info
    
    def _stack_history(self):
        return np.stack(list(self.buffer), axis=0).astype(np.float32)
    
    def _maybe_trigger_scheduled_event(self):
        if self._schedule_idx >= len(self._schedule):
            return
        ev = self._schedule[self._schedule_idx]
        if self.steps == ev["step"]:
            self._apply_event(ev["type"], ev["params"], scheduled=True)
            self._schedule_idx += 1
            self._last_event_step = self.steps
    
    def _maybe_trigger_random_event(self):
        if self.steps - self._last_event_step < self.event_min_gap:
            return
        if self.rng.rand() >= self.event_prob:
            return
        roll = self.rng.rand()
        if roll < 1/3:
            self._apply_event("deadline_shock", {}, scheduled=False)
        elif roll < 2/3:
            self._apply_event("speed_surge", {}, scheduled=False)
        else:
            self._apply_event("energy_drought", {}, scheduled=False)
        self._last_event_step = self.steps
    
    def _apply_event(self, etype: str, params: dict, scheduled: bool):
        info = {"type": etype, "step": self.steps, "scheduled": scheduled}
        
        if etype == "deadline_shock":
            shrink_abs = int(params.get("shrink_abs", 0))
            shrink_ratio = float(params.get("shrink_ratio", self.default_deadline_shock_ratio))
            if shrink_abs > 0:
                shrink = min(self.deadline - 1, shrink_abs)
            else:
                shrink = int(np.ceil(max(0, self.deadline) * shrink_ratio))
            self.deadline = max(1, self.deadline - shrink)
            info.update({"shrink": shrink})
            self.event_id = 'deadline_shock'
        
        elif etype == "speed_surge":
            mult = float(params.get("mult", self.default_speed_mult))
            self.scales['speed'] *= mult
            info.update({"speed_scale": self.scales['speed'], "mult": mult})
            self.event_id = 'speed_surge'
        
        elif etype == "energy_drought":
            mult = float(params.get("mult", self.default_energy_mult))
            self.scales['energy'] *= mult
            info.update({"energy_scale": self.scales['energy'], "mult": mult})
            self.event_id = 'energy_drought'
        
        else:
            raise ValueError(f"Unknown event type: {etype}")

        self.phase += 1
        self._pending_event = info
        self.info['event_id'] = self.event_id
        self.info['phase'] = self.phase
    
    def render(self):
        if hasattr(self.base_env, 'render'):
            return self.base_env.render()
        return None
    
    def close(self):
        if hasattr(self.base_env, 'close'):
            self.base_env.close()


if __name__ == '__main__':
    env = HalfCheetahEnv(render_mode='rgb_array')
    obs, info = env.reset()
    terminated = truncated = False
    total = 0.0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
    print(total)
    print(info)
    env.close()
