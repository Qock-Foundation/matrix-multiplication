import sys
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gen_samples(n, m, p, n_samples):
    A = torch.normal(mean=0., std=5., size=(n_samples, n, m))
    B = torch.normal(mean=0., std=5., size=(n_samples, m, p))
    Y = A @ B
    A = A.reshape((n_samples, -1))
    B = B.reshape((n_samples, -1))
    X = torch.cat((A, B), dim=1)
    return X, Y


class MatmulModel(nn.Module):
    def __init__(self, n, m, p, k):
        super().__init__()
        self.n = n
        self.m = m
        self.p = p
        self.k = k
        self.layer1 = nn.Linear(n * m + m * p, 2 * k, bias=False)
        self.layer3 = nn.Linear(k, n * p, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = x[:, :self.k] * x[:, self.k:]
        x = self.layer3(x)
        return x


def train_and_test(model, optimizer, criterion, train_loader, test_loader, num_epochs, tol, alpha):
    train_losses, test_losses = [], []
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            loss_old = loss.item()
            for param in model.parameters():
                loss += alpha * torch.sum((param - torch.round(param)) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss += loss_old * len(X_batch)
        train_losses.append(train_loss / len(train_loader.dataset))

        model.eval()
        test_loss = 0
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            with torch.no_grad():
                preds = model(X_batch)
                loss = criterion(preds, Y_batch)
                test_loss += loss.item() * len(X_batch)
        test_losses.append(test_loss / len(test_loader.dataset))
        if test_losses[-1] < tol:
            break
    return train_losses, test_losses


def plot_losses(train_losses, test_losses):
    plt.yscale('log')
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.legend()


def output_algorithm(model):
    with torch.no_grad():
        n, m, p, k = model.n, model.m, model.p, model.k
        C1, C2 = [param for param in model.parameters()]
        for t in range(2 * k):
            print(f'y_{t} = ', end='')
            for ind in range(n * m):
                print(f'{round(C1[t][ind].item(), 3)} * A_{ind // m}{ind % m} + ', end='')
            for ind in range(m * p):
                print(f'{round(C1[t][n * m + ind].item(), 3)} * B_{ind // p}{ind % p} + ', end='')
            print('0')
        for t in range(k):
            print(f'z_{t} = y_{t} * y_{t + k}')
        for ind in range(n * p):
            print(f'C_{ind // p}{ind % p} = ', end='')
            for t in range(k):
                print(f'{round(C2[ind][t].item(), 3)} * z_{t} + ', end='')
            print('0')


n = int(sys.argv[1])
m = int(sys.argv[2])
p = int(sys.argv[3])
k = int(sys.argv[4])
lr = float(sys.argv[5])

X, Y = gen_samples(n, m, p, 1024)
X = X.reshape((len(X), -1)).float()
Y = Y.reshape((len(Y), -1)).float()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

tol = 1e-4
for t in range(1, 101):
    model = MatmulModel(n, m, p, k).to(device)
    optimizer = torch.optim.ASGD(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=128, shuffle=False)
    train_losses, test_losses = train_and_test(model, optimizer, nn.MSELoss(), train_loader, test_loader,
                                               num_epochs=20000, tol=tol, alpha=1e-4)
    print(f'Attempt {t}: train_loss = {train_losses[-1]}, test_loss = {test_losses[-1]}')
    if test_losses[-1] < tol:
        output_algorithm(model)
        break

# plot_losses(train_losses, test_losses)
