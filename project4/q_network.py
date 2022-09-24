import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools


class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        # --------- YOUR CODE HERE --------------
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 9)
        pass
        # ---------------------------------------

    def forward(self, x, device):
        # --------- YOUR CODE HERE --------------
        x = torch.FloatTensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        # ---------------------------------------

    def select_discrete_action(self, obs, device):
        # Put the observation through the network to estimate q values for all possible discrete actions
        est_q_vals = self.forward(obs.reshape((1,) + obs.shape), device)
        # Choose the discrete action with the highest estimated q value
        discrete_action = torch.argmax(est_q_vals, dim=1).tolist()[0]
        return discrete_action

    def action_discrete_to_continuous(self, discrete_action):
        # --------- YOUR CODE HERE --------------
        actions = {}
        get_actions(actions)
        # print(actions)
        # print(len(actions))
        return actions[discrete_action]
        # ---------------------------------------


def get_actions(actions):
    j = 0.75
    actions[0] = np.array([[j], [0]])  # ([[0.01], [0.285]])
    actions[1] = np.array([[j], [j]])  # ([[-0.05], [-0.18]])
    actions[2] = np.array([[j], [-j]])  # ([[-0.25], [0.4]])
    actions[3] = np.array([[0], [-j]])  # ([[-0.06], [-0.30]])
    actions[4] = np.array([[0], [j]])   # ([[0.2], [0.38]])
    actions[5] = np.array([[0], [0]])  # ([[0], [0.25]])
    actions[6] = np.array([[-j], [j]])
    actions[7] = np.array([[-j], [0]])
    actions[8] = np.array([[-j], [-j]])
    # actions[9] = np.array([[0.1], [j]])
    # actions[10] = np.array([[-0.1], [j]])
    # actions[11] = np.array([[0.1], [-j]])
    # actions[12] = np.array([[-0.1], [-j]])
    # actions[13] = np.array([[0], [-2 * j]])
    # actions[14] = np.array([[0], [2 * j]])
    return actions
