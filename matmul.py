import sys
import torch
from copy import deepcopy
from torch import nn
from tqdm import trange

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def custom_round(x, d):
    return torch.round(x * d) / d


class MatmulModel(nn.Module):
    def __init__(self, n, m, p, k):
        super().__init__()
        self.n, self.m, self.p, self.k = n, m, p, k
        self.layer1 = nn.Linear(n * m + m * p, 2 * k, bias=False)
        self.layer2 = nn.Linear(k, n * p, bias=False)
        self.fixed1 = torch.zeros_like(self.layer1.weight, dtype=torch.bool)

    def forward(self, x):
        x = self.layer1(x)
        x = x[:, :self.k] * x[:, self.k:]
        x = self.layer2(x)
        return x

    def order_params(self, d):
        params = torch.flatten(self.layer1.weight)
        fixed = torch.flatten(self.fixed1)
        delta = torch.abs(params - custom_round(params, d)) + fixed
        return torch.argsort(delta)[:sum(~fixed)]

    def round_distance(self, ind, d):
        params = torch.flatten(self.layer1.weight)
        return torch.abs(params[ind] - custom_round(params[ind], d))

    @torch.no_grad()
    def round_param(self, ind, d):
        m = self.layer1.weight.shape[1]
        i, j = ind // m, ind % m
        assert not self.fixed1[i][j]
        self.fixed1[i][j] = True
        self.layer1.weight[i][j] = custom_round(self.layer1.weight[i][j], d)

    def zero_grad_fixed(self):
        self.layer1.weight.grad[self.fixed1] = 0


def attempt(model, optimizer, scheduler, criterion, num_batches, batch_size, tol):
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
        if loss.isnan():
            return False
        if loss < tol:
            return True
        optimizer.zero_grad()
        loss.backward()
        model.zero_grad_fixed()
        optimizer.step()
        scheduler.step()
    return False


def stringify_monom(c, sym):
    if c == 0:
        return '0'
    if c == 1:
        return sym
    if c == -1:
        return '-' + sym
    return str(round(c, 3)) + ' * ' + sym


def stringify_linear(cs, syms):
    indices = torch.nonzero(cs != 0)
    return ' + '.join([stringify_monom(cs[i].item(), syms[i]) for i in indices])


@torch.no_grad()
def print_algorithm(model):
    n, m, p, k = model.n, model.m, model.p, model.k
    c1, c2 = [param for param in model.parameters()]
    syms = \
        [f'A_{ind // m}{ind % m}' for ind in range(n * m)] + \
        [f'B_{ind // p}{ind % p}' for ind in range(m * p)]
    for t in range(2 * k):
        print(f'y_{t} = {stringify_linear(c1[t], syms)}')
    for t in range(k):
        print(f'z_{t} = y_{t} * y_{t + k}')
    syms = [f'z_{t}' for t in range(k)]
    for ind in range(n * p):
        print(f'C_{ind // p}{ind % p} = {stringify_linear(c2[ind], syms)}')


def main(n, m, p, k, lr):
    num_batches = 400000
    num_batches2 = 100000
    batch_size = 8
    tol = 1e-4
    max_d = 6
    while True:
        model = MatmulModel(n, m, p, k).to(device)
        optimizer = torch.optim.ASGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        if attempt(model, optimizer, scheduler, nn.MSELoss(),
                   num_batches=num_batches, batch_size=batch_size, tol=tol):
            break
    for d in range(1, max_d + 1):
        for ind in model.order_params(d):
            num_params = len(torch.flatten(model.fixed1))
            num_fixed = sum(torch.flatten(model.fixed1))
            delta = model.round_distance(ind, d)
            print(f'Progress: {num_fixed}/{num_params}, delta = {delta}, d = {d}', file=sys.stderr)
            model_new = deepcopy(model)
            model_new.round_param(ind, d)
            optimizer = torch.optim.ASGD(model_new.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
            if attempt(model_new, optimizer, scheduler, nn.MSELoss(),
                       num_batches=num_batches2, batch_size=batch_size, tol=tol):
                model = model_new
    if not model.fixed1.all():
        print('Not converged :(', file=sys.stderr)
    print_algorithm(model)


if __name__ == '__main__':
    main(n=int(sys.argv[1]),
         m=int(sys.argv[2]),
         p=int(sys.argv[3]),
         k=int(sys.argv[4]),
         lr=float(sys.argv[5]))
