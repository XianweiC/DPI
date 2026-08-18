from tqdm import tqdm

import torch

from DPI.env import QueueEnv, MazeEnv, HalfCheetahEnv
from DPI.agent.dpi_q import DPI_Q
from DPI.agent.dpi_ppo import DPI_PPO
from DPI.core.trainer import train
from DPI.run.argparser import get_args
from DPI.misc.utils import setup_system
from DPI.misc.rollout_buffer import RolloutBuffer
from DPI.misc.rollout_buffer import queue_rollout, maze_rollout, halfcheetah_rollout


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 2025 + args.seed_id
    setup_system(seed)

    if args.env_name == 'Queue-v0':
        env = QueueEnv()
        input_size = env.observation_space.shape
        output_size = env.action_space.n
        reward_size = 5
        network_type = 'queue'
        action_type = 'discrete'
    elif args.env_name == 'Maze-v0':
        env = MazeEnv()
        input_size = env.observation_space.shape
        output_size = env.action_space.n
        reward_size = 5
        network_type = 'maze'
        action_type = 'discrete'
    elif args.env_name == 'Half-Cheetah-v0':
        env = HalfCheetahEnv()
        input_size = env.observation_space.shape
        output_size = env.action_space.shape[0]
        reward_size = env.d
        network_type = 'half-cheetah'
        action_type = 'continuous'
    observation_space = env.observation_space
    obs, info = env.reset(seed=seed)

    print("-"*60)
    print(f'env name: {args.env_name}')
    print(f'observation space: {observation_space}')
    print(f'action space: {output_size}')
    print("-"*60)


    TOTAL_TIMESTEPS = int(args.max_step)
    ROLLOUT_LENGTH = args.rollout_length
    BATCH_SIZE = args.batch_size
    EPOCH = args.epoch
    NUM_ENVS = args.num_envs
    K = 8
    LR = args.learning_rate

    agent = DPI_PPO(observation_space, output_size, 128, reward_size, network_type, action_type, LR)

    print("-"*60)
    print(f'network type: {network_type}')
    print("-"*60)

    steps_done = 0
    pbar = tqdm(range(0, TOTAL_TIMESTEPS, ROLLOUT_LENGTH), desc="Training")
    for step in pbar:
        rollout_buffer = RolloutBuffer(ROLLOUT_LENGTH, NUM_ENVS, input_size, output_size, reward_size)
        if args.env_name == 'Queue-v0':
            rollout_buffer = queue_rollout(env, agent, rollout_buffer, ROLLOUT_LENGTH, K, device)
        elif args.env_name == 'Maze-v0':
            rollout_buffer = maze_rollout(env, agent, rollout_buffer, ROLLOUT_LENGTH, K, device)
        elif args.env_name == 'Half-Cheetah-v0':
            rollout_buffer = halfcheetah_rollout(env, agent, rollout_buffer, ROLLOUT_LENGTH, K, device)
        train(agent, rollout_buffer, EPOCH, BATCH_SIZE, device=device)
        steps_done += ROLLOUT_LENGTH * NUM_ENVS
        pbar.set_postfix(steps=steps_done)
