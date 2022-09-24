from base import RobotPolicy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import load_data
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


class RGBBCRobot2(RobotPolicy):
    def __init__(self):
        self.layers = [3, 6, 16, 120, 32, 4]
        self.network = MyDNN(self.layers)
        self.trainer = MyDNNTrain(self.network)
        print('Layers: ', self.layers)

    def train(self, data):
        #data = load_data('./data/map1.pkl')
        #data.pop('poses')
        #data.pop('agent')
        #data['obs'] = data.pop('rgb')

        features = torch.tensor(data['obs'])
        features = features.permute((0, 3, 1, 2))
        features = np.asarray(features)
        labels = np.asarray(data['actions']).reshape(-1, 1)
        self.trainer.train(labels, features)
        pass

    def get_action(self, obs):
        obs = np.reshape(obs, (1, 64, 64, 3))
        obs = torch.tensor(obs)
        obs = obs.permute((0, 3, 1, 2))
        obs = np.asarray(obs)
        return self.network.predict(obs)

class MyDNN(nn.Module):
    def __init__(self, layers):
        super(MyDNN, self).__init__()
        self.conv1 = nn.Conv2d(layers[0], layers[1], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(layers[1], layers[2], 5)
        self.fc1 = nn.Linear(2704, layers[3])
        self.fc2 = nn.Linear(layers[3], layers[4])
        self.fc3 = nn.Linear(layers[4], layers[5])

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, features):
        self.eval()
        features = torch.from_numpy(features).float()
        y = self.forward(features)
        _, y_pred = torch.max(y, 1)
        return y_pred

class MyDataset(Dataset):
    def __init__(self, labels, features):
        super(MyDataset, self).__init__()
        self.labels = labels
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return {'feature': feature, 'label': label}

class MyDNNTrain(object):
    def __init__(self, network):
        self.network = network
        self.learning_rate = 0.01
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 100
        self.batchsize = 50
        self.shuffle = True

    def train(self, labels, features):
        self.network.train()
        dataset = MyDataset(labels, features)
        loader = DataLoader(dataset, shuffle=self.shuffle, batch_size=self.batchsize)
        for epoch in range(self.num_epochs):
            self.train_epoch(loader, epoch)

    def train_epoch(self, loader, epoch):
        total_loss = 0.0
        for i, data in enumerate(loader):
            #print(i, data)
            features = data['feature'].float()
            labels = data['label'].long()
            labels = labels.squeeze(1)
            self.optimizer.zero_grad()
            predictions = self.network(features)
            loss = self.criterion(predictions, labels)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
        if (epoch % 5 == 0) or (epoch == self.num_epochs - 1):
            print('At epoch:', epoch, ' Loss=', total_loss/i)
