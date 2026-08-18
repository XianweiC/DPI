from collections import deque
from typing import Dict, Tuple, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def build_layout(
    gh: int,
    gw: int,
    corridor_width: int = 1,
    hazard_band_width: int = 1,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    margin: int = 1,
    s_stride: int = 2, 
    top_margin: int = 1,
    bottom_margin: int = 7, 
) -> Dict:
    """
    - short_path: 贴边走（先底边到最右列，再最右列向上）
    - long_path : S型蛇形，逐列推进到最右列
    """
    assert gh >= 7 and gw >= 7, "at least 7x7 grid for two seprate paths."
    assert s_stride >= 1

    if start is None:
        start = (gh - 1, 0)      
    if goal is None:
        goal = (0, gw - 1)       

    walls = np.ones((gh, gw), dtype=np.uint8)
    static_hazard = np.zeros((gh, gw), dtype=np.uint8)

    def carve_polyline(cells: List[Tuple[int,int]]):
        for (r, c) in cells:
            walls[r, c] = 0

    short_path_cells: List[Tuple[int,int]] = []
    r, c = start

    while c < gw - 1:
        short_path_cells.append((r, c))
        c += 1
    while r > goal[0]:
        short_path_cells.append((r, c))
        r -= 1
    short_path_cells.append(goal)
    carve_polyline(short_path_cells)

    static_hazard[2:, -1] = 1
    static_hazard[-1, 1:] = 1

    walls[:, 0] = 0
    walls[0, :-5] = 0
    walls[1, -2] = 0
    
    long_path_cells: List[Tuple[int,int]] = []

    cur_col = gw - 5
    last_col = gw - 2

    start_r = 0

    direction_down = True
    while cur_col < last_col:
        r_top = max(0 + top_margin, 0)
        r_bot = min(gh - 1 - bottom_margin, gh - 1)
        
        if direction_down:
            while start_r < r_bot - 1:
                long_path_cells.append((start_r, cur_col))
                start_r += 1
        else:
            while start_r > r_top:
                long_path_cells.append((start_r, cur_col))
                start_r -= 1

        next_col = min(cur_col + s_stride, last_col)
        while cur_col < next_col:
            long_path_cells.append((start_r, cur_col))
            cur_col += 1

        direction_down = not direction_down
        
    carve_polyline(long_path_cells)

    return {
        'walls': walls,
        'static_hazard': static_hazard,
        'start': tuple(start),
        'goal':  tuple(goal),
        'short_path_cells': short_path_cells,
        'long_path_cells':  long_path_cells,
    }


def draw_rect(canvas, top, left, height, width, value):
    H, W = canvas.shape
    t = max(0, int(top)); l = max(0, int(left))
    b = min(H, int(top + height)); r = min(W, int(left + width))
    if b > t and r > l:
        canvas[t:b, l:r] = value


class MazeEnv(gym.Env):
    """
    Discrete 2D navigation with non-stationary hazards, deadline, and energy.

    Observation: stacked history (H_win, C, H, W)
      C channels:
        0: hazard heatmap (静态高危 + 动态风暴融合)
        1: wall
        2: agent mask
        3: goal mask
        4: time (top)
        5: energy (bottom)

    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=WAIT

    Vector reward (d=5):
      [progress_shaping, time_penalty, hazard_penalty, energy_penalty, deadline_objective]

    Events (piecewise non-stationarity):
      info['event_id'] in { 'none','deadline_shock','hazard_surge','energy_drought','goal_move' }
      info['phase']    in { 1, 2, 3, ... }
    """
    metadata = {"render_modes": ["tensor", "rgb_array"], "name": "Maze-v1"}

    def __init__(self,
                 grid_h=11, grid_w=11,
                 history_size=8,
                 max_steps=200,
                 base_deadline=120,
                 goal_bonus=20.0,
                 hazard_scale=4.0,
                 energy_move_cost=0.6,
                 energy_wait_cost=0.2,
                 energy_init=30.0,
                 deadline_bonus=15.0,
                 storm_speed=0.15,
                 seed: Optional[int]=None,
                 schedule: Optional[List[Tuple[int, str]]] = None,
                 render_mode: Optional[str]=None):
        super().__init__()
        self.render_mode = render_mode

        self.H, self.W = 84, 84
        self.gh, self.gw = grid_h, grid_w
        self.H_win = int(history_size)

        self.max_steps = max_steps
        self.base_deadline0 = int(base_deadline)
        self.goal_bonus0 = float(goal_bonus)
        self.hazard_scale0 = float(hazard_scale)
        self.energy_move_cost0 = float(energy_move_cost)
        self.energy_wait_cost0 = float(energy_wait_cost)
        self.energy_init0 = float(energy_init)
        self.deadline_bonus0 = float(deadline_bonus)
        self.storm_speed0 = float(storm_speed)

        self.rng = np.random.RandomState(seed)

        self.C = 6
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.H_win, self.C, self.gh, self.gw), 
            dtype=np.float32
        )

        self._base_scales = {
            "progress": 1.0,
            "time_penalty": 1.0,
            "hazard_penalty": 1.0,
            "energy_penalty": 1.0,
            "deadline_bonus": 1.0,
        }
        self.scales = self._base_scales.copy()
        self.default_schedule = schedule
        self.layout = None

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict]=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.scales = self._base_scales.copy()

        self.layout = None
        if options:
            self.layout = options.get('layout', None)
        if options and 'schedule' in options and options['schedule'] is not None:
            self.schedule = list(options['schedule'])
        elif self.default_schedule is not None:
            self.schedule = list(self.default_schedule)
        else: # defalut
            self.schedule = [(40, 'deadline_shock'),
                             (80, 'hazard_surge'),
                             (120, 'energy_drought')]

        self.steps = 0
        self.deadline = int(self.base_deadline0)
        self.energy = float(self.energy_init0)
        self.done = False
        self.reached = False
        self.phase = 1
        self.event_id = 'none'

        if self.layout is not None:
            self.agent = np.array(self.layout['start'])
            self.goal  = np.array(self.layout['goal'])
        else:
            self.agent = np.array([self.rng.randint(0, self.gh), self.rng.randint(0, self.gw)])
            self.goal  = np.array([self.rng.randint(0, self.gh), self.rng.randint(0, self.gw)])
            while (self.goal == self.agent).all():
                self.goal = np.array([self.rng.randint(0, self.gh), self.rng.randint(0, self.gw)])

        self.storm_center = np.array([self.rng.uniform(0, self.gh-1),
                                      self.rng.uniform(0, self.gw-1)], dtype=np.float32)
        v = self.rng.randn(2).astype(np.float32); v /= (np.linalg.norm(v) + 1e-8)
        self.storm_dir = v
        self.storm_speed = self.storm_speed0
        self.hazard_scale = self.hazard_scale0
        self.sigma2 = 3.5

        self.energy_move_cost = self.energy_move_cost0
        self.energy_wait_cost = self.energy_wait_cost0
        self.goal_bonus = self.goal_bonus0
        self.deadline_bonus = self.deadline_bonus0
        self.base_deadline = self.base_deadline0

        self.prev_dist = self._manhattan(self.agent, self.goal)

        self.info = {
            'time': self.deadline,
            'life': 1,
            'score': 0.0,
            'reached': False,
            'energy': self.energy,
            'deadline': self.deadline,
            'storm_center': self.storm_center.copy(),
            'mor': np.zeros(5, dtype=np.float32),
            'event_id': self.event_id,
            'phase': self.phase
        }

        # first_frame = self._make_frame()
        first_frame = self._make_tensor_frame()
        self.buffer = deque([first_frame.copy() for _ in range(self.H_win)], maxlen=self.H_win)

        obs = self._stack_history()
        return obs, self.info

    def get_evt_time(self):
        res = []
        for t, evt in self.schedule:
            res.append(t)
        return res

    def _is_wall(self, pos) -> bool:
        if self.layout is None:
            return False
        r, c = int(pos[0]), int(pos[1])
        return bool(self.layout['walls'][r, c] == 1)

    def _static_hazard(self, pos) -> float:
        if self.layout is None:
            return 0.0
        r, c = int(pos[0]), int(pos[1])
        return float(self.layout['static_hazard'][r, c])

    def _manhattan(self, a, b) -> int:
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    # -------------- 事件脚本 --------------
    def _apply_events_if_any(self):
        triggered = []
        for (t, name) in list(self.schedule):
            if self.steps == t:
                triggered.append((t, name))
                self.schedule.remove((t, name))
        if not triggered:
            return

        self.phase += 1
        ev = 'none'
        for (_, name) in triggered:
            if name == 'deadline_shock': # 选择最短路径
                # 剩余时间砍半 + 提高与时限相关项
                self.deadline = max(0, int(self.deadline * 0.5))
                self.scales['time_penalty'] = 1.5
                self.scales['deadline_bonus'] = 1.5
                ev = 'deadline_shock'
            elif name == 'hazard_surge': # 尽可能避开
                # 风暴更快/更强/更尖锐
                self.storm_speed *= 1.6
                self.hazard_scale *= 1.4
                self.sigma2 = 2.5
                ev = 'hazard_surge'
            elif name == 'energy_drought':
                # 能耗上升
                self.energy_move_cost *= 1.5
                self.energy_wait_cost *= 1.5
                self.scales['energy_penalty'] = 1.3
                ev = 'energy_drought'
        self.event_id = ev
        self.info['event_id'] = self.event_id
        self.info['phase'] = self.phase

    # -------------- Step --------------
    def step(self, action: int):
        # 返回: (obs, reward_scalar, terminated, truncated, info)
        if self.done:
            return self._stack_history(), 0.0, True, False, self.info

        self.steps += 1
        self._apply_events_if_any()

        # 风暴移动 + 反弹
        self.storm_center += self.storm_dir * self.storm_speed
        for i, lim in enumerate([self.gh-1, self.gw-1]):
            if self.storm_center[i] < 0 or self.storm_center[i] > lim:
                self.storm_dir[i] *= -1
                self.storm_center[i] = np.clip(self.storm_center[i], 0, lim)

        # 动作与能耗
        move = np.array([0, 0])
        if action == 0:   # UP
            move = np.array([-1, 0]); self.energy -= self.energy_move_cost
        elif action == 1: # DOWN
            move = np.array([1, 0]);  self.energy -= self.energy_move_cost
        elif action == 2: # LEFT
            move = np.array([0, -1]); self.energy -= self.energy_move_cost
        elif action == 3: # RIGHT
            move = np.array([0, 1]);  self.energy -= self.energy_move_cost
        elif action == 4: # WAIT
            move = np.array([0, 0]);  self.energy -= self.energy_wait_cost

        nxt = self.agent + move
        nxt[0] = int(np.clip(nxt[0], 0, self.gh-1))
        nxt[1] = int(np.clip(nxt[1], 0, self.gw-1))
        if self._is_wall(nxt):
            # 碰墙：不移动（可在此处加轻微惩罚）
            nxt = self.agent.copy()
        self.agent = nxt

        self.deadline -= 1

        haz = self._hazard_intensity(self.agent)

        # 进度 shaping：靠近则给势能差
        dist = self._manhattan(self.agent, self.goal)
        progress = float(max(0, self.prev_dist - dist))
        if (self.agent == self.goal).all():
            self.reached = True
            progress += self.goal_bonus
        self.prev_dist = dist

        time_pen = -1.0
        hazard_pen = -self.hazard_scale * haz
        energy_pen = - (self.energy_move_cost if (move != 0).any() else self.energy_wait_cost)
        deadline_obj = 0.0

        # 终止条件
        reached_deadline = self.deadline <= 0
        out_of_energy = self.energy <= 0
        terminated = False
        truncated = False
        if self.reached or reached_deadline or out_of_energy:
            terminated = True
            if self.reached and not reached_deadline and not out_of_energy:
                deadline_obj = self.deadline_bonus
            elif reached_deadline and not self.reached:
                deadline_obj = -self.deadline_bonus
            if out_of_energy and not self.reached:
                energy_pen += -2.0  # NOTE: reconfig...
        elif self.steps >= self.max_steps:
            truncated = True

        mor = np.array([
            self.scales['progress'] * progress,
            self.scales['time_penalty'] * time_pen,
            self.scales['hazard_penalty'] * hazard_pen,
            self.scales['energy_penalty'] * energy_pen,
            self.scales['deadline_bonus'] * deadline_obj,
        ], dtype=np.float32)

        scalar_r = float(np.clip(mor.sum(), -50.0, 50.0))
        score = float(10.0 * self.reached - 5.0 * reached_deadline - 2.0 * haz)

        # info
        self.info.update({
            'score': score,
            'reached': self.reached,
            'energy': float(self.energy),
            'deadline': max(0, int(self.deadline)),
            'storm_center': self.storm_center.copy(),
            'mor': mor,
            'event_id': self.event_id,
            'phase': self.phase
        })

        if terminated or truncated:
            self.done = True

        # 帧缓冲更新
        # self.buffer.append(self._make_frame())
        self.buffer.append(self._make_tensor_frame())
        obs = self._stack_history()
        return obs, scalar_r, terminated, truncated, self.info

    # -------------- 危害强度 --------------
    def _hazard_intensity(self, pos) -> float:
        if isinstance(pos, (tuple, list, np.ndarray)):
            r, c = int(pos[0]), int(pos[1])
        else:
            r, c = pos
        # 静态高危（短路必险）
        base = self._static_hazard((r, c))  # 0 或 1
        # 动态风暴（高斯核）
        dy = float(r) - float(self.storm_center[0])
        dx = float(c) - float(self.storm_center[1])
        d2 = (dx*dx + dy*dy)
        storm = np.exp(-d2 / (2.0 * self.sigma2))
        # 融合：用 max 确保静态高危不被稀释
        return float(max(base, storm))
        # return float(storm)

    # -------------- 观测帧构造 --------------
    def _make_frame(self):
        H, W, C = self.H, self.W, self.C
        layers = np.zeros((C, H, W), dtype=np.float32)

        margin = 6
        cell_h = (H - 2*margin) // self.gh
        cell_w = (W - 2*margin) // self.gw
        y0 = margin; x0 = margin

        # 0) hazard heatmap
        # 1) wall mask
        haz_layer = layers[0]
        wall_layer = layers[1]
        for r in range(self.gh):
            for c in range(self.gw):
                haz = self._hazard_intensity((r, c))
                y = y0 + r*cell_h; x = x0 + c*cell_w
                draw_rect(haz_layer, y, x, cell_h-2, cell_w-2, 0.15 + 0.6*haz)

                wall = self._is_wall((r, c))
                draw_rect(wall_layer, y, x, cell_h-2, cell_w-2, wall)

        # 1) agent mask
        ay = y0 + int(self.agent[0])*cell_h
        ax = x0 + int(self.agent[1])*cell_w
        draw_rect(layers[2], ay+2, ax+2, max(1, cell_h-6), max(1, cell_w-6), 1.0)

        # 2) goal mask
        gy = y0 + int(self.goal[0])*cell_h
        gx = x0 + int(self.goal[1])*cell_w
        draw_rect(layers[3], gy+2, gx+2, max(1, cell_h-6), max(1, cell_w-6), 0.9)

        # 3) time bar（剩余时间 / 初始时限）
        frac_t = max(0.0, min(1.0, self.deadline / max(1, self.base_deadline)))
        draw_rect(layers[4], 1, margin, 3, int(frac_t * (W - 2*margin)), 0.9)

        # 4) energy bar（剩余能量 / 初始能量）
        frac_e = max(0.0, min(1.0, self.energy / max(1.0, self.energy_init0)))
        draw_rect(layers[5], H-6, margin, 3, int(frac_e * (W - 2*margin)), 0.75)

        return layers

    # -------------- 观测帧构造 --------------
    def _make_tensor_frame(self):
        H, W, C = self.gh, self.gw, self.C
        layers = np.zeros((C, H, W), dtype=np.float32)

        # 0) hazard heatmap
        # 1) wall mask
        for r in range(self.gh):
            for c in range(self.gw):
                # layers[0][r, c] = 0.15 + 0.6*self._hazard_intensity((r, c))
                layers[0][r, c] = self._hazard_intensity((r, c))
                layers[1][r, c] = 1 if self._is_wall((r, c)) else 0

        # 1) agent mask
        ay, ax = int(self.agent[0]), int(self.agent[1])
        if 0 <= ay < self.gh and 0 <= ax < self.gw:
            layers[2, ay, ax] = 1

        # 2) goal mask
        gy, gx = int(self.goal[0]), int(self.goal[1])
        if 0 <= gy < self.gh and 0 <= gx < self.gw:
            layers[3, gy, gx] = 1.0

        # 4) time ratio（全图常数）
        # 避免除零：base_deadline 可能被事件修改，但这里用初始 base 归一化更稳定
        time_ratio = float(np.clip(self.deadline / max(1, self.base_deadline0), 0.0, 1.0))
        layers[4, :, :] = time_ratio

        # 5) energy ratio（全图常数）
        energy_ratio = float(np.clip(self.energy / max(1.0, self.energy_init0), 0.0, 1.0))
        layers[5, :, :] = energy_ratio

        return layers

    def _stack_history(self):
        return np.stack(list(self.buffer), axis=0).astype(np.float32)

    def render(self):
        return self._make_frame().sum(0)
        # return self._make_tensor_frame().sum(0)


if __name__ == "__main__":
    layout = build_layout(
        gh=11, gw=11, corridor_width=1, hazard_band_width=1, margin=1
    )

    env = MazeEnv(grid_h=11, grid_w=11, history_size=8, render_mode="rgb_array")
    obs, info = env.reset(options={
        'layout': layout,
        'schedule': [(40, 'deadline_shock'), (80, 'hazard_surge'), (120, 'energy_drought')]
    })
    terminated = truncated = False
    total = 0.0

    import matplotlib.pyplot as plt; import time
    plt.ion()  # 打开交互模式, 可以持续可视化
    fig, ax = plt.subplots()
    img = ax.imshow(env.render(), 
                    cmap='jet'
                    )  # 初始化图像

    # while not (terminated or truncated):
    #     a = np.random.randint(0, 5)
    #     obs, r, terminated, truncated, info = env.step(a)
    #     total += r
    #     # 可视化帧（rgb）
    #     # img = env.render()

    #     img.set_data(env.render())  # 更新图像
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()
    #     # time.sleep(0.05)

    # print("Episode return:", total, "info:", {k: info[k] for k in ['reached','event_id','phase']})

    trans_table = {'w': 0, 's': 1, 'a': 2, 'd':3}
    for i in range(600000):
        # action = env.action_space.sample()
        action = input('please input:'); action = trans_table[action]
        # action = int(action); print(action)
        obs, scalar_r, terminated, truncated, info = env.step(action)

        # print(obs.shape)
        print(info['energy'], info['deadline'], info['mor'])
        # for i in range(8):
        #     plt.subplot(1, 8, i+1)
        #     plt.imshow(obs[i].mean(0), cmap='jet')

        img.set_data(env.render())  # 更新图像
        fig.canvas.draw()
        fig.canvas.flush_events()
        # time.sleep(0.05)

        done = np.logical_or(terminated, truncated)
        if done:
            obs, info = env.reset(options={
                'layout': layout,
                'schedule': [(40, 'deadline_shock'), 
                            (80, 'hazard_surge'), 
                            (120, 'energy_drought')]
                })
    # break
    del fig, ax
