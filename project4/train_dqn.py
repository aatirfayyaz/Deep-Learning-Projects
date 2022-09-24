import argparse
import numpy as np
import time
import random
import os
import signal
from contextlib import contextmanager
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
# import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer
from q_network import QNetwork
from arm_env import ArmEnv


# ---------- Utils for setting time constraints -----#
class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# ---------- End of timing utils -----------#


class TrainDQN:

    @staticmethod
    def add_arguments(parser):
        # Common arguments
        parser.add_argument('--learning_rate', type=float, default=7e-4,
                            help='the learning rate of the optimizer')
        # LEAVE AS DEFAULT THE SEED YOU WANT TO BE GRADED WITH
        parser.add_argument('--seed', type=int, default=17,
                            help='seed of the experiment')
        parser.add_argument('--save_dir', type=str, default='models',
                            help="the root folder for saving the checkpoints")
        parser.add_argument('--gui', action='store_true', default=False,
                            help="whether to turn on GUI or not")
        # 7 minutes by default
        parser.add_argument('--time_limit', type=int, default=7 * 60,
                            help='time limits for running the training in seconds')

    def __init__(self, env, device, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        self.env = env
        self.env.seed(args.seed)
        self.env.observation_space.seed(args.seed)
        self.device = device
        self.q_network = QNetwork(env).to(self.device)
        print(self.device.__repr__())
        print(self.q_network)

    def save_model(self, episode_num, episode_reward, args):
        model_folder_name = f'episode_{episode_num:06d}_reward_{round(episode_reward):03d}'
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(self.q_network.state_dict(), os.path.join(args.save_dir, model_folder_name, 'q_network.pth'))
        print(f'Model saved to {os.path.join(args.save_dir, model_folder_name, "q_network.pth")}\n')

    def train(self, args):
        # --------- YOUR CODE HERE --------------

        learning_rate = args.learning_rate
        target_network = self.q_network
        capacity = 5000
        num_episodes = 205
        min_experience = 10
        batch_size = 128
        replay_buffer = ReplayBuffer(capacity)
        episode_lim = 5
        discount_factor = 0.99
        num_actions = 9
        best_reward = -999
        # max_grad = 0.5
        end_time = 200
        episode_rewards, network_losses = [], []
        best_episode = 0

        optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        mse_loss = nn.MSELoss()

        for episode in range(num_episodes):
            episode_reward = 0
            observation = self.env.reset()
            epsilon = 0.7
            network_loss = 0.0

            for t in range(end_time):
                if episode > min_experience:
                    epsilon = 0.01 + 0.69 * math.exp(-(episode - min_experience) / 200)

                if np.random.rand() < epsilon:
                    discrete_action = np.random.randint(0, num_actions)
                    action = self.q_network.action_discrete_to_continuous(discrete_action)
                    # print('Random action taken')
                else:
                    discrete_action = self.q_network.select_discrete_action(observation, self.device)
                    action = self.q_network.action_discrete_to_continuous(discrete_action)
                    # print('Best action taken')

                new_observation, reward, done, _ = self.env.step(action)

                if done:
                    break

                transition = observation, discrete_action, reward, new_observation, done
                episode_reward += reward
                replay_buffer.put(transition)

                if episode > min_experience:

                    s_lst, a_lst, r_lst, s_prime_lst, _ = replay_buffer.sample(batch_size)

                    observations = torch.from_numpy(s_lst).float()
                    new_observations = torch.from_numpy(s_prime_lst).float()
                    actions = torch.from_numpy(a_lst)
                    rewards = torch.from_numpy(r_lst).float()

                    X = self.q_network.forward(observations, self.device).gather(1, actions.unsqueeze(-1))

                    Y = torch.max(target_network.forward(new_observations, self.device).detach(), dim=1)[0].detach()
                    Y = (rewards + discount_factor * Y).unsqueeze(1)

                    loss = mse_loss(X, Y)

                    optimizer.zero_grad()
                    loss.backward()
                    # nn.utils.clip_grad_norm_(list(self.q_network.parameters()), max_grad)
                    optimizer.step()
                    network_loss += loss.item()

                observation = new_observation

            print(f"Episode reward: {episode_reward:.02f} @ episode number: {episode} and epsilon: {epsilon:.03f} "
                  f"with Network Loss: {network_loss:.04f}")

            network_losses.append(network_loss)
            episode_rewards.append(episode_reward)
            avg_reward = np.average(np.array(episode_rewards[-7:]))
            # print(avg_reward, best_reward)

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_episode = episode
            # print(best_episode)
            self.save_model(episode, episode_reward, args)

            if episode % episode_lim == 0:
                target_network.load_state_dict(self.q_network.state_dict())

        # Plots for network performance:
        print('Best episode was: ', best_episode)

        # t = np.arange(0, num_episodes, 1)
        # plt.subplot(121)
        # plt.plot(t, episode_rewards)
        # plt.xlabel('Episode Number')
        # plt.ylabel('Episode Reward')
        # plt.subplot(122)
        # plt.plot(t, network_losses)
        # plt.xlabel('Episode Number')
        # plt.ylabel('Network Loss')
        # plt.suptitle('Network Performance')
        # plt.show(block=False)
        # plt.show()

        pass
        # ---------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    TrainDQN.add_arguments(parser)
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    if not args.seed: args.seed = int(time.time())

    env = ArmEnv(args)
    device = torch.device('cpu')
    # declare Q function network and set seed
    tdqn = TrainDQN(env, device, args)
    # run training under time limit
    try:
        with time_limit(args.time_limit):
            tdqn.train(args)
    except TimeoutException as e:
        print("You ran out of time and your training is stopped!")
