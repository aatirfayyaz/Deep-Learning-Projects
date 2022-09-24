import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.optim as optim
import argparse
import time
from models import *

np.set_printoptions(suppress=True)

np.random.seed(0)
torch.manual_seed(0)

class DynamicDataset(Dataset):
    def __init__(self, dataset_dir):
        # X: (N, 9), Y: (N, 6)
        self.X = np.load(os.path.join(dataset_dir, 'X.npy')).T.astype(np.float32)
        self.Y = np.load(os.path.join(dataset_dir, 'Y.npy')).T.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# class Net(nn.Module):
#     # ---
#     # Your code goes here
#
#     pass
#     # ---


def train(model, train_loader, epoch):
    model.train()

    # ---
    # Your code goes here
    learning_rate = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)
    criterion = nn.MSELoss()

    total = 0
    train_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        X, labels = data
        optimizer.zero_grad()
        Y = model(X)
        loss = criterion(Y, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += labels.size(0)
    print('Epoch #:', epoch + 1, '  has a training loss of: ', train_loss / total)

    # ---


def test(model, test_loader):
    model.eval()

    # --
    # Your code goes here
    criterion = nn.MSELoss()

    test_loss = 0.0
    total = 0
    for i, data in enumerate(test_loader, 0):
        X, labels = data
        Y = model(X.float())
        loss = criterion(Y.float(), labels.float())

        test_loss += loss.item()
        total += labels.size(0)

    test_loss = test_loss / total
    print('Testing loss calculated as: ', test_loss)
    # ---

    return test_loss


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--time_step', type=float, default=0.01)
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    return args


def main():
    args = get_args()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = DynamicDataset(args.dataset_dir)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set)
    test_loader = torch.utils.data.DataLoader(test_set)

    # ---
    # Your code goes here

    model = build_model(args.num_links, args.time_step)
    print('Starting training')
    best_loss = 1000000
    best_epoch = -1
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        train(model, train_loader, epoch)
        test_loss = test(model, test_loader)

        if test_loss < best_loss:
            print('Better model found!')
            best_loss = test_loss
            best_epoch = epoch

        model_folder_name = f'epoch_{epoch:04d}_loss_{test_loss:.8f}'
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(model.state_dict(), os.path.join(args.save_dir, model_folder_name, 'dynamics.pth'))
        print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "dynamics.pth")}\n')

        print('Training complete')
        print('Best Epoch: ', best_epoch, '  Best Loss :', best_loss)
        print('\n\n')

    # ---


if __name__ == '__main__':
    main()
