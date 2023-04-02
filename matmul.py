import sys
import torch
from torch import nn
from tqdm import trange

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MatmulModel(nn.Module):
    def __init__(self, n, m, p, k):
        super().__init__()
        self.n, self.m, self.p, self.k = n, m, p, k
        self.layer1 = nn.Linear(n * m + m * p, 2 * k, bias=False)
        self.layer3 = nn.Linear(k, n * p, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = x[:, :self.k] * x[:, self.k:]
        x = self.layer3(x)
        return x


def attempt(model, optimizer, criterion, num_batches, batch_size, tol, alpha):
    model.train()
    pbar = trange(num_batches, unit='batches')
    for _ in pbar:
        a = 5 * torch.randn(batch_size, model.n, model.m)
        b = 5 * torch.randn(batch_size, model.m, model.p)
        x = torch.cat((a.reshape(batch_size, -1), b.reshape(batch_size, -1)), dim=1)
        y = (a @ b).reshape(batch_size, -1)
        z = model(x)
        loss = criterion(y, z)
        pbar.set_postfix(loss=f'{loss.item():.6f}', refresh=False)
        if loss < tol:
            return True
        for param in model.parameters():
            loss += alpha * torch.sum((param - torch.round(param)) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return False


def print_algorithm(model):
    with torch.no_grad():
        n, m, p, k = model.n, model.m, model.p, model.k
        c1, c2 = [param for param in model.parameters()]
        for t in range(2 * k):
            print(f'y_{t} = ', end='')
            for ind in range(n * m):
                print(f'{round(c1[t][ind].item(), 3)} * A_{ind // m}{ind % m} + ', end='')
            for ind in range(m * p):
                print(f'{round(c1[t][n * m + ind].item(), 3)} * B_{ind // p}{ind % p} + ', end='')
            print('0')
        for t in range(k):
            print(f'z_{t} = y_{t} * y_{t + k}')
        for ind in range(n * p):
            print(f'C_{ind // p}{ind % p} = ', end='')
            for t in range(k):
                print(f'{round(c2[ind][t].item(), 3)} * z_{t} + ', end='')
            print('0')


def main(n, m, p, k, lr):
    while True:
        model = MatmulModel(n, m, p, k).to(device)
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr)
        if attempt(model, optimizer, nn.MSELoss(),
                   num_batches=200000, batch_size=128, tol=1e-4, alpha=1e-4):
            print_algorithm(model)
            break


if __name__ == '__main__':
    main(n=int(sys.argv[1]),
         m=int(sys.argv[2]),
         p=int(sys.argv[3]),
         k=int(sys.argv[4]),
         lr=float(sys.argv[5]))
