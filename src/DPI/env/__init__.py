import gymnasium as gym

from .queue_env import QueueEnv
from .maze_env import MazeEnv
from .half_cheetah_env import HalfCheetahEnv

__all__ = ['QueueEnv', 'MazeEnv', 'HalfCheetahEnv']

gym.register(id="Queue-v0", entry_point=QueueEnv, max_episode_steps=1000)
gym.register(id="Maze-v0", entry_point=MazeEnv, max_episode_steps=1000)
gym.register(id="HalfCheetah-v0", entry_point=HalfCheetahEnv, max_episode_steps=1000)
