import sys
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_matrices(n, n_samples):
    a = torch.normal(mean=0., std=5., size=(n_samples, n, n))
    b = torch.normal(mean=0., std=5., size=(n_samples, n, n))
    y = a @ b
    a = a.reshape((n_samples, -1))
    b = b.reshape((n_samples, -1))
    x = torch.cat((a, b), dim=1)
    return x, y


class MatmulModel(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.layer1 = nn.Linear(2 * n ** 2, 2 * k, bias=False)
        self.layer3 = nn.Linear(k, n ** 2, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = x[:, :k] * x[:, k:]
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
        C1, C2 = [param for param in model.parameters()]
        for t in range(2 * k):
            print(f'y_{t} = ', end='')
            for ind in range(n ** 2):
                print(f'{round(C1[t][ind].item(), 3)} * A_{ind // n}{ind % n} + ', end='')
            for ind in range(n ** 2):
                print(f'{round(C1[t][n ** 2 + ind].item(), 3)} * B_{ind // n}{ind % n} + ', end='')
            print('0')
        for t in range(k):
            print(f'z_{t} = y_{t} * y_{t + k}')
        for ind in range(n ** 2):
            print(f'C_{ind // n}{ind % n} = ', end='')
            for t in range(k):
                print(f'{round(C2[ind][t].item(), 3)} * z_{t} + ', end='')
            print('0')


n = int(sys.argv[1])
X, Y = generate_matrices(n, 1024)
X = X.reshape((len(X), -1)).float()
Y = Y.reshape((len(Y), -1)).float()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

k = int(sys.argv[2])
tol = 1e-4
for t in range(1, 101):
    model = MatmulModel(n, k).to(device)
    optimizer = torch.optim.ASGD(model.parameters(), lr=1e-3)
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=128, shuffle=False)
    train_losses, test_losses = train_and_test(model, optimizer, nn.MSELoss(), train_loader, test_loader,
                                               num_epochs=20000, tol=tol, alpha=1e-4)
    print(f'Attempt {t}: train_loss = {train_losses[-1]}, test_loss = {test_losses[-1]}')
    if test_losses[-1] < tol:
        output_algorithm(model)
        break

# plot_losses(train_losses, test_losses)
