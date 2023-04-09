import torch
from copy import deepcopy
from torch import nn
from tqdm import trange

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


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
        self.fixed1 = torch.full_like(self.layer1.weight, torch.nan).to(device)

    def forward(self, x):
        x = self.layer1(x)
        x = x[:, :self.r] * x[:, self.r:]
        x = self.layer2(x)
        return x

    def gen_sample(self, batch_size, scale):
        a = scale * torch.randn(batch_size, self.n, self.m)
        b = scale * torch.randn(batch_size, self.m, self.p)
        x = torch.cat((a.reshape(batch_size, -1), b.reshape(batch_size, -1)), dim=1).to(device)
        y = (a @ b).reshape(batch_size, -1).to(device)
        return x, y

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
        self.fixed11 = torch.full_like(self.layer11.weight, torch.nan).to(device)
        self.fixed12 = torch.full_like(self.layer12.weight, torch.nan).to(device)

    def forward(self, x):
        x1 = self.layer11(x[:, :self.n * self.m])
        x2 = self.layer12(x[:, self.n * self.m:])
        x = self.layer2(x1 * x2)
        return x

    def gen_sample(self, batch_size, scale):
        a = scale * torch.randn(batch_size, self.n, self.m)
        b = scale * torch.randn(batch_size, self.m, self.p)
        x = torch.cat((a.reshape(batch_size, -1), b.reshape(batch_size, -1)), dim=1).to(device)
        y = (a @ b).reshape(batch_size, -1).to(device)
        return x, y

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


class TransposeMatmulModel(nn.Module):
    def __init__(self, n, m, r):
        super().__init__()
        self.n, self.m, self.r = n, m, r
        self.layer1 = nn.Linear(n * m, 2 * r, bias=False)
        self.layer2 = nn.Linear(r, n * n, bias=False)
        self.fixed1 = torch.full_like(self.layer1.weight, torch.nan).to(device)

    def forward(self, x):
        x = self.layer1(x)
        x = x[:, :self.r] * x[:, self.r:]
        x = self.layer2(x)
        return x

    def gen_sample(self, batch_size, scale):
        a = scale * torch.randn(batch_size, self.n, self.m)
        x = a.reshape(batch_size, -1).to(device)
        y = (a @ a.transpose(1, 2)).reshape(batch_size, -1).to(device)
        return x, y

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
        syms = [f'A_{ind // self.m}{ind % self.m}' for ind in range(self.n * self.m)]
        for t in range(2 * self.r):
            print(f'y_{t} = {stringify_linear(self.layer1.weight[t], syms)}')
        for t in range(self.r):
            print(f'z_{t} = y_{t} * y_{t + self.r}')
        syms = [f'z_{t}' for t in range(self.r)]
        for ind in range(self.n * self.n):
            print(f'C_{ind // self.n}{ind % self.n} = {stringify_linear(self.layer2.weight[ind], syms)}')


def attempt(model, optimizer, scheduler, criterion, num_batches, batch_size, scale, tol):
    model.train()
    pbar = trange(num_batches, unit='batches')
    for _ in pbar:
        x, y = model.gen_sample(batch_size, scale)
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


def solve(model_creator):
    num_batches1 = 400000
    num_batches2 = 300000
    batch_size1 = 1024
    batch_size2 = 256
    gamma = 1 - 3 / num_batches1
    lr1 = 1e-3
    lr2 = 2e-4
    lr3 = 1e-6
    tol1 = 1e-2
    tol3 = 1e-6
    scale = 2
    denominators = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12]

    print('Approximation stage')
    while True:
        model = model_creator().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        if attempt(model, optimizer, scheduler, loss_function,
                   num_batches=num_batches1, batch_size=batch_size1, scale=scale, tol=tol1):
            break
    model.output()

    print('Separation stage')
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
                print(f'[{num_fixed + 1}/{num_params}] separating {old_value} -> {new_value}')
                model_new = deepcopy(model)
                model_new.fix_param(ind, new_value)
                optimizer = torch.optim.Adam(model_new.parameters(), lr=lr2)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
                if attempt(model_new, optimizer, scheduler, loss_function,
                           num_batches=num_batches2, batch_size=batch_size2, scale=scale, tol=tol1):
                    model = model_new
                    break
            else:
                break

    print('Refinement stage')
    model_final = deepcopy(model)
    optimizer = torch.optim.Adam(model_final.parameters(), lr=lr3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    if attempt(model_final, optimizer, scheduler, loss_function,
               num_batches=num_batches1, batch_size=batch_size1, scale=scale, tol=tol3):
        model_final.output()
    else:
        print('Failed to rationalize :(')
        model.output()
