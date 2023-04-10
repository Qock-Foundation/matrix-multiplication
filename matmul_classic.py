import sys
import torch
from torch import nn
from utils import stringify_linear
from solve import solve


class ClassicMatmulModel(nn.Module):
    def __init__(self, n, m, p, r):
        super().__init__()
        self.n, self.m, self.p, self.r = n, m, p, r
        self.layer11 = nn.Linear(n * m, r, bias=False)
        self.layer12 = nn.Linear(m * p, r, bias=False)
        self.layer2 = nn.Linear(r, n * p, bias=False)

    def __str__(self):
        result = ''
        syms = [f'A_{i // self.m}{i % self.m}' for i in range(self.n * self.m)]
        for t in range(self.r):
            result += f'y_{t} = {stringify_linear(self.layer11.weight[t], syms)}\n'
        syms = [f'B_{i // self.p}{i % self.p}' for i in range(self.m * self.p)]
        for t in range(self.r):
            result += f'y_{t + self.r} = {stringify_linear(self.layer12.weight[t], syms)}\n'
        for t in range(self.r):
            result += f'z_{t} = y_{t} * y_{t + self.r}\n'
        syms = [f'z_{t}' for t in range(self.r)]
        for i in range(self.n * self.p):
            result += f'C_{i // self.p}{i % self.p} = {stringify_linear(self.layer2.weight[i], syms)}\n'
        return result

    def forward(self, x):
        x1 = self.layer11(x[:, :self.n * self.m])
        x2 = self.layer12(x[:, self.n * self.m:])
        x = self.layer2(x1 * x2)
        return x

    def sample(self, batch_size, scale):
        a = scale * torch.randn(batch_size, self.n, self.m)
        b = scale * torch.randn(batch_size, self.m, self.p)
        x = torch.cat((a.reshape(batch_size, -1), b.reshape(batch_size, -1)), dim=1)
        y = (a @ b).reshape(batch_size, -1)
        return x, y


n = int(sys.argv[1])
m = int(sys.argv[2])
p = int(sys.argv[3])
r = int(sys.argv[4])
solve(lambda: ClassicMatmulModel(n, m, p, r))
