import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self, num_links, time_step):
        super().__init__()
        self.num_links = num_links
        self.time_step = time_step

    def forward(self, x):
        qddot = self.compute_qddot(x)
        state = x[:, :2 * self.num_links]
        next_state = self.compute_next_state(state, qddot)
        return next_state

    def compute_next_state(self, state, qddot):
        qdot_old = state[:, self.num_links:2 * self.num_links]
        qdot_new = qdot_old + qddot * torch.tensor(self.time_step)
        q_old = state[:, :self.num_links]
        q_new = q_old + torch.tensor(0.5) * (qdot_old + qdot_new) * torch.tensor(self.time_step)
        next_state = torch.cat([q_new, qdot_new], dim=1)
        return next_state

    def compute_qddot(self, x):
        pass


class Model1Link(Model):
    def __init__(self, time_step):
        super().__init__(1, time_step)
        # Your code goes here
        # Using powers of 2
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def compute_qddot(self, x):
        # Your code goes here
        pass


class Model2Link(Model):
    def __init__(self, time_step):
        super().__init__(1, time_step)
        # Your code goes here
        # Using powers of 2
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 18)
        self.fc5 = nn.Linear(18, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def compute_qddot(self, x):
        # Your code goes here
        pass


class Model3Link(Model):
    def __init__(self, time_step):
        super().__init__(1, time_step)
        # Your code goes here
        # Using powers of 2
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def compute_qddot(self, x):
        # Your code goes here
        pass


def build_model(num_links, time_step):
    if num_links == 1:
        model = Model1Link(time_step)
    elif num_links == 2:
        model = Model2Link(time_step)
    elif num_links == 3:
        model = Model3Link(time_step)
    return model
