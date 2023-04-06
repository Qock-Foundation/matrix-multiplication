import sys
import torch
from copy import deepcopy
from torch import nn
from tqdm import trange

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def loss_function(x, y):
    return torch.sqrt(torch.mean((x - y) ** 2))


def custom_round(x, d):
    return torch.round(x * d) / d


def round_distance(x, d):
    return torch.abs(x - custom_round(x, d))


def stringify_monom(c, sym, eps=1e-2):
    if abs(c) < eps:
        return '0'
    if abs(c - 1) < eps:
        return sym
    if abs(c + 1) < eps:
        return '-' + sym
    return str(round(c, 5)) + ' * ' + sym


def stringify_linear(cs, syms, eps=1e-2):
    mask = torch.nonzero(torch.abs(cs) > eps)
    return ' + '.join([stringify_monom(cs[i].item(), syms[i], eps) for i in mask])


class CommMatmulModel(nn.Module):
    def __init__(self, n, m, p, r):
        super().__init__()
        self.n, self.m, self.p, self.r = n, m, p, r
        self.layer1 = nn.Linear(n * m + m * p, 2 * r, bias=False)
        self.layer2 = nn.Linear(r, n * p, bias=False)
        self.fixed1 = torch.full_like(self.layer1.weight, torch.nan)

    def forward(self, x):
        x = self.layer1(x)
        x = x[:, :self.r] * x[:, self.r:]
        x = self.layer2(x)
        return x

    def get_params(self):
        return torch.flatten(self.layer1.weight)

    def get_fixed(self):
        return ~torch.flatten(self.fixed1).isnan()

    @torch.no_grad()
    def fix_param(self, ind, value):
        m = self.layer1.weight.shape[1]
        i, j = ind // m, ind % m
        self.layer1.weight[i][j] = value
        self.fixed1[i][j] = value

    @torch.no_grad()
    def reset_fixed(self):
        mask1 = ~self.fixed1.isnan()
        self.layer1.weight[mask1] = self.fixed1[mask1]

    @torch.no_grad()
    def output(self):
        syms = \
            [f'A_{ind // self.m}{ind % self.m}' for ind in range(self.n * self.m)] + \
            [f'B_{ind // self.p}{ind % self.p}' for ind in range(self.m * self.p)]
        for t in range(2 * self.r):
            print(f'y_{t} = {stringify_linear(self.layer1.weight[t], syms)}')
        for t in range(self.r):
            print(f'z_{t} = y_{t} * y_{t + self.r}')
        syms = [f'z_{t}' for t in range(self.r)]
        for ind in range(self.n * self.p):
            print(f'C_{ind // self.p}{ind % self.p} = {stringify_linear(self.layer2.weight[ind], syms)}')


class ClassicMatmulModel(nn.Module):
    def __init__(self, n, m, p, r):
        super().__init__()
        self.n, self.m, self.p, self.r = n, m, p, r
        self.layer11 = nn.Linear(n * m, r, bias=False)
        self.layer12 = nn.Linear(m * p, r, bias=False)
        self.layer2 = nn.Linear(r, n * p, bias=False)
        self.fixed11 = torch.full_like(self.layer11.weight, torch.nan)
        self.fixed12 = torch.full_like(self.layer12.weight, torch.nan)

    def forward(self, x):
        x1 = self.layer11(x[:, :self.n * self.m])
        x2 = self.layer12(x[:, self.n * self.m:])
        x = self.layer2(x1 * x2)
        return x

    def get_params(self):
        return torch.cat((torch.flatten(self.layer11.weight), torch.flatten(self.layer12.weight)))

    def get_fixed(self):
        return ~torch.cat((torch.flatten(self.fixed11), torch.flatten(self.fixed12))).isnan()

    @torch.no_grad()
    def fix_param(self, ind, value):
        size1 = self.layer11.weight.shape[0] * self.layer11.weight.shape[1]
        m1, m2 = self.layer11.weight.shape[1], self.layer12.weight.shape[1]
        if ind < size1:
            i1, j1 = ind // m1, ind % m1
            self.layer11.weight[i1][j1] = value
            self.fixed11[i1][j1] = value
        else:
            i2, j2 = (ind - size1) // m2, (ind - size1) % m2
            self.layer12.weight[i2][j2] = value
            self.fixed12[i2][j2] = value

    @torch.no_grad()
    def reset_fixed(self):
        mask11 = ~self.fixed11.isnan()
        self.layer11.weight[mask11] = self.fixed11[mask11]
        mask12 = ~self.fixed12.isnan()
        self.layer12.weight[mask12] = self.fixed12[mask12]

    @torch.no_grad()
    def output(self):
        syms = [f'A_{ind // self.m}{ind % self.m}' for ind in range(self.n * self.m)]
        for t in range(self.r):
            print(f'y_{t} = {stringify_linear(self.layer11.weight[t], syms)}')
        syms = [f'B_{ind // self.p}{ind % self.p}' for ind in range(self.m * self.p)]
        for t in range(self.r):
            print(f'y_{t + self.r} = {stringify_linear(self.layer12.weight[t], syms)}')
        for t in range(self.r):
            print(f'z_{t} = y_{t} * y_{t + self.r}')
        syms = [f'z_{t}' for t in range(self.r)]
        for ind in range(self.n * self.p):
            print(f'C_{ind // self.p}{ind % self.p} = {stringify_linear(self.layer2.weight[ind], syms)}')


def attempt(model, optimizer, scheduler, criterion, num_batches, batch_size, scale, tol):
    model.train()
    pbar = trange(num_batches, unit='batches')
    for _ in pbar:
        a = scale * torch.randn(batch_size, model.n, model.m)
        b = scale * torch.randn(batch_size, model.m, model.p)
        x = torch.cat((a.reshape(batch_size, -1), b.reshape(batch_size, -1)), dim=1).to(device)
        y = (a @ b).reshape(batch_size, -1).to(device)
        z = model(x)
        loss = criterion(y, z)
        pbar.set_postfix(loss=f'{loss.item():.6f}', refresh=False)
        if loss.isnan():
            return False
        if loss < tol:
            return True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.reset_fixed()
    return False


def main(tp, n, m, p, r):
    model_class = CommMatmulModel if tp == 'comm' else ClassicMatmulModel
    num_batches = 100000
    num_batches2 = 100000
    batch_size = 1024
    batch_size2 = 128
    gamma = 1 - 3 / num_batches
    gamma2 = 1 - 3 / num_batches2
    lr = 1e-3
    lr2 = 1e-3
    tol = 1e-2
    tol2 = 1e-12
    scale = 2
    denominators = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12]
    while True:
        model = model_class(n, m, p, r).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        if attempt(model, optimizer, scheduler, loss_function,
                   num_batches=num_batches, batch_size=batch_size, scale=scale, tol=tol):
            break
    model.output()
    for d in denominators:
        used = torch.zeros_like(model.get_params(), dtype=torch.bool)
        while True:
            for ind in torch.argsort(round_distance(model.get_params(), d)):
                if used[ind] or model.get_fixed()[ind]:
                    continue
                used[ind] = True
                num_params = len(model.get_params())
                num_fixed = sum(model.get_fixed())
                old_value = model.get_params()[ind]
                new_value = custom_round(old_value, d)
                print(f'[{num_fixed + 1}/{num_params}] rounding {old_value} -> {new_value}')
                model_new = deepcopy(model)
                model_new.fix_param(ind, new_value)
                optimizer = torch.optim.ASGD(model_new.parameters(), lr=lr2)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma2)
                if attempt(model_new, optimizer, scheduler, loss_function,
                           num_batches=num_batches2, batch_size=batch_size2, scale=scale, tol=tol):
                    model = model_new
                    break
            else:
                break
    print('Final optimization')
    model_final = deepcopy(model)
    optimizer = torch.optim.Adam(model_final.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    if attempt(model_final, optimizer, scheduler, nn.MSELoss(),
               num_batches=num_batches, batch_size=batch_size, scale=scale, tol=tol2):
        model_final.output()
    else:
        print('Failed to converge :(')
        model.output()


if __name__ == '__main__':
    main(tp=sys.argv[1],
         n=int(sys.argv[2]),
         m=int(sys.argv[3]),
         p=int(sys.argv[4]),
         r=int(sys.argv[5]))
