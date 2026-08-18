import numpy as np
from collections import deque

import gymnasium as gym
from gymnasium import spaces


class QueueEnv(gym.Env):
    """
    QueueEnv with vectorized rewards, dynamic events (random + scripted),
    and morality budget as a hard constraint.

    Reward vector per step:
      mor = [progress, time_pen, fairness_pen, energy_pen, deadline_obj]

    Success:
      success = served and deadline>0 and energy>0 and morality_score>morality_min

    Events (three types):
      - "deadline_shock" : shorten remaining deadline (urgency up)
      - "morality_surge" : multiply fairness/morality pressure (norm pressure up)
      - "energy_drought" : multiply energy costs (scarcity up)
    """
    metadata = {"render_modes": ["ansi"], "name": "Queue-v0"}

    def __init__(self,
                 history_size=8,
                 max_queue_len=50,
                 base_deadline=10,          
                 max_steps=10,
                 service_rate=1,
                 max_service_rate=3,
                 cut_k=1,
                 goal_bonus=20.0,
                 deadline_bonus=15.0,
                 morality_scale=3.0,        
                 energy_init=10.0,          
                 wait_cost=0.5,             
                 cut_cost=1.0,              
                 
                 morality_init=1.0,         
                 morality_min=0.0,          
                 morality_step=0.1,         
                 
                 enable_events=True,
                 event_min_gap=20,          
                 event_prob=0.12,           
                 deadline_shock_ratio=0.5,  
                 morality_mult=2.0,         
                 energy_mult=2.0,           
                 
                
                 schedule_only=False,       
                 
                 return_vector=False,
                 seed=None,
                 
                 adapt_deadline_to_pos=True,
                 extra_deadline_buffer=5,
                 ):
        super().__init__()

        
        self.H_win = int(history_size)
        self.max_queue_len = int(max_queue_len)
        self.base_deadline = int(base_deadline)
        self.max_steps = int(max_steps)

        self.service_rate = int(service_rate)
        self.max_service_rate = int(max_service_rate)
        self.cut_k = int(cut_k)

        self.goal_bonus = float(goal_bonus)
        self.deadline_bonus = float(deadline_bonus)
        self.morality_scale_base = float(morality_scale)  

        self.energy_init = float(energy_init)
        self.wait_cost_base = float(wait_cost)
        self.cut_cost_base = float(cut_cost)

        
        self.morality_init = float(morality_init)
        self.morality_min = float(morality_min)
        self.morality_step = float(morality_step)

        
        self.enable_events = bool(enable_events)
        self.event_min_gap = int(event_min_gap)
        self.event_prob = float(event_prob)
        self.default_deadline_shock_ratio = float(deadline_shock_ratio)
        self.default_morality_mult = float(morality_mult)
        self.default_energy_mult = float(energy_mult)

        
        self.schedule_only = bool(schedule_only)

        
        self.return_vector = bool(return_vector)

        
        self.rng = np.random.RandomState(seed)

        
        self.F = 6  
        self.d = 5  

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.H_win, self.F), 
                                            dtype=np.float32)

        
        self.scales = {"morality": 1.0, "energy": 1.0}

        
        self.adapt_deadline_to_pos = bool(adapt_deadline_to_pos)
        self.extra_deadline_buffer = int(extra_deadline_buffer)

        self.reset()

    

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        self._schedule_raw = []
        schedule = None if options is None else options.get("schedule")
        self._schedule_raw = [] if schedule is None else list(schedule)

        
        qlen = self.rng.randint(self.max_queue_len // 2, self.max_queue_len + 1)
        
        pos = self.rng.randint(1, qlen + 1)  

        self.queue_len = int(qlen)
        self.pos = int(pos)
        self.deadline = int(self.base_deadline)
        self.energy = float(self.energy_init)
        self.morality_score = float(self.morality_init)
        self.steps = 0
        self.done = False
        self.served = False

        self.scales["morality"] = 1.0
        self.scales["energy"] = 1.0

        
        if self.adapt_deadline_to_pos:
            
            min_steps_needed = int(np.ceil(self.pos / max(1, self.service_rate)))
            self.deadline = max(10, min(self.base_deadline, min_steps_needed + self.extra_deadline_buffer))

        
        self._last_event_step = -10**9
        self._pending_event = None

        
        self._schedule = []
        for item in self._schedule_raw:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                raise ValueError("event_schedule item must be (etype, step[, params])")
            etype, step = item[0], int(item[1])
            params = dict(item[2]) if len(item) >= 3 and item[2] is not None else {}
            self._schedule.append({"type": str(etype), "step": step, "params": params})
        self._schedule.sort(key=lambda e: e["step"])
        self._schedule_idx = 0

        
        first_feat = self._make_feature(recent_cut=0)
        self.buffer = deque([first_feat.copy() for _ in range(self.H_win)], maxlen=self.H_win)

        
        self.info = {
            "served": False,
            "queue_len": self.queue_len,
            "pos": self.pos,
            "energy": self.energy,
            "deadline": self.deadline,
            "morality_score": self.morality_score,
            "mor": np.zeros(self.d, dtype=np.float32),
            "reward_names": ["progress", "time_pen", "fairness_pen", "energy_pen", "deadline_obj"],
            "event": None,
            "term_cause": None,
            "success": 0
        }

        obs = self._stack_history()
        return obs, self.info

    def get_evt_time(self):
        res = []
        for item in self._schedule:
            res.append(item['step'])
        return res

    def step(self, action: int):
        if self.done:
            empty = np.zeros(self.d, np.float32)
            return self._stack_history(), (empty if self.return_vector else 0.0), True, False, self.info

        self.steps += 1
        recent_cut = 0
        self.info.pop("event", None)
        self._pending_event = None

        
        prev_pos = self.pos
        wait_cost = self.wait_cost_base * self.scales["energy"]
        cut_cost  = self.cut_cost_base  * self.scales["energy"]

        if action == 0:  
            self.energy -= wait_cost
        elif action == 1:  
            self.pos = max(0, self.pos - self.cut_k)
            self.energy -= cut_cost
            
            self.morality_score -= self.morality_step * self.scales["morality"]
            recent_cut = 1
        else:
            raise ValueError("Invalid action")

        progress = 0.0
        if max(0, prev_pos - self.pos) > 0:
            progress += 1.0  

        
        if self.pos == 0 and not self.served:
            self.served = True
            progress += self.goal_bonus
            self.queue_len = max(0, self.queue_len - 1)

        
        if not self.served:
            served_now = min(self.service_rate, self.queue_len)
            self.queue_len -= served_now
            self.pos = max(0, self.pos - served_now)
            if self.pos == 0 and self.queue_len >= 0 and not self.served:
                self.served = True
                progress += self.goal_bonus

        
        self.deadline -= 1

        
        self._maybe_trigger_scheduled_event()
        
        if self.enable_events and not self.schedule_only:
            self._maybe_trigger_random_event()

        
        time_pen = -1.0
        fairness_pen = - self.morality_scale_base * self.scales["morality"] * recent_cut * self.cut_k
        energy_pen = - (cut_cost if action == 1 else wait_cost)
        deadline_obj = 0.0

        
        timeout = self.deadline <= 0
        exhausted = self.energy <= 0.0
        immoral = self.morality_score <= self.morality_min

        terminated = False
        truncated = False
        term_cause = None

        if self.served or timeout or exhausted or immoral:
            terminated = True
            if self.served and not timeout and not exhausted and not immoral:
                deadline_obj = self.deadline_bonus
                term_cause = "served"
            elif timeout and not self.served:
                deadline_obj = -self.deadline_bonus
                term_cause = "timeout"
            elif exhausted and not self.served:
                energy_pen += -2.0  
                term_cause = "exhausted"
            elif immoral and not self.served:
                
                
                term_cause = "immoral"

        if self.steps >= self.max_steps and not terminated:
            truncated = True
            term_cause = "truncated"

        
        success_flag = int(self.served and (not timeout) and (not exhausted) and (not immoral))

        mor = np.array([progress, time_pen, fairness_pen, energy_pen, deadline_obj], dtype=np.float32)
        scalar_r = float(np.clip(mor.sum(), -50.0, 50.0))

        
        self.info.update({
            
            "served": self.served,
            "queue_len": self.queue_len,
            "pos": self.pos,
            "energy": float(self.energy),
            "deadline": max(0, self.deadline),
            "morality_score": float(self.morality_score),
            "mor": mor,
            "reward_names": ["progress", "time_pen", "fairness_pen", "energy_pen", "deadline_obj"],
            "event": self._pending_event,
            "term_cause": term_cause,
            "success": success_flag
        })

        self.buffer.append(self._make_feature(recent_cut=recent_cut))
        obs = self._stack_history()
        self.done = terminated or truncated

        reward_out = mor if self.return_vector else scalar_r
        return obs, reward_out, terminated, truncated, self.info

    def render(self):
        status = (f"(t={self.steps}) pos={self.pos} len={self.queue_len} "
                  f"deadline={self.deadline} energy={self.energy:.1f} served={self.served} "
                  f"morality={self.morality_score:.2f} sr={self.service_rate} scales={self.scales}")
        print(status)
        return status

    

    def _make_feature(self, recent_cut: int):
        """Return a normalized feature vector in [0,1] for the current state."""
        pos_norm = 1.0 - np.clip(self.pos / max(1, self.max_queue_len), 0.0, 1.0)  
        qlen_norm = np.clip(self.queue_len / max(1, self.max_queue_len), 0.0, 1.0)
        energy_norm = np.clip(self.energy / max(1.0, self.energy_init), 0.0, 1.0)
        deadline_norm = np.clip(self.deadline / max(1, self.base_deadline), 0.0, 1.0)
        cut_flag = float(1 if recent_cut else 0)
        sr_norm = np.clip(self.service_rate / max(1, self.max_service_rate), 0.0, 1.0)
        return np.array([pos_norm, qlen_norm, energy_norm, deadline_norm, cut_flag, sr_norm], dtype=np.float32)

    def _stack_history(self):
        return np.stack(list(self.buffer), axis=0).astype(np.float32)

    

    def _maybe_trigger_scheduled_event(self):
        if getattr(self, "_schedule_idx", 0) >= len(getattr(self, "_schedule", [])):
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
            self._apply_event("morality_surge", {}, scheduled=False)
        else:
            self._apply_event("energy_drought", {}, scheduled=False)
        self._last_event_step = self.steps

    

    def _apply_event(self, etype: str, params: dict, scheduled: bool):
        """Apply a specific event; params can override defaults."""
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

        elif etype == "morality_surge":
            mult = float(params.get("mult", self.default_morality_mult))
            self.scales["morality"] *= mult
            info.update({"morality_scale": self.scales["morality"], "mult": mult})

        elif etype == "energy_drought":
            mult = float(params.get("mult", self.default_energy_mult))
            self.scales["energy"] *= mult
            info.update({"energy_scale": self.scales["energy"], "mult": mult})

        else:
            raise ValueError(f"Unknown event type: {etype}")

        self._pending_event = info
