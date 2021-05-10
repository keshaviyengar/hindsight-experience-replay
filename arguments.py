import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='CTR-Reach-v0', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=int(5e4), help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=10, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=50, help='the times to update the network')
    # 2. - not used
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    # 4. - not used
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    # Not used
    parser.add_argument('--noise-eps', type=float, default=0.7, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e5), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    # 6. - Update to clip based on spaces
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.95, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.0005, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.0005, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.001, help='the average coefficient. Same as Tau.')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    # 10.
    parser.add_argument('--clip-range', type=float, default=None, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=1, help='the rollouts per mpi')
    parser.add_argument('--normalize-inputs', action='store_true', help='normalize states and actions.')

    args = parser.parse_args()

    return args
