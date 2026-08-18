import argparse

def get_args():
    parser = argparse.ArgumentParser(description='DPI')
    # parser.add_argument('--env-name', default='Queue-v0', choices=['Queue-v0', 'Maze-v0', 'Half-Cheetah-v0'], help='experiment environment name.')
    # parser.add_argument('--env-name', default='Maze-v0', choices=['Queue-v0', 'Maze-v0', 'Half-Cheetah-v0'], help='experiment environment name.')
    parser.add_argument('--env-name', default='Half-Cheetah-v0', choices=['Queue-v0', 'Maze-v0', 'Half-Cheetah-v0'], help='experiment environment name.')

    # running configuration
    parser.add_argument('--seed-id', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--training', action='store_true', help='run for training (default FALSE)')
    parser.add_argument('--render', action='store_true', help='render the game (default FALSE)')
    parser.add_argument('--num-envs', type=int, default=1, metavar='NWORKER', help='number of parallel environments (defualt 32)')
    # hyperparameters
    parser.add_argument('--max-step', type=int, default=1.5e5, metavar='MSTEP', help='max number of steps for learning rate scheduling (default 1.15e8)')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help='initial learning rate (default 2.5e-4)')
    parser.add_argument('--reward-scale', type=float, default=1.0, help='reward scaling (default 1.0)')
    parser.add_argument('--batch-size', type=int, default=1024, help='number of batch size used for one backward updating.')
    parser.add_argument('--epoch', type=int, default=1, help='number of epoch using one episode for training.')
    parser.add_argument('--rollout-length', type=int, default=4096, help='number of rollout length in one episode.')

    args = parser.parse_args()
    return args
