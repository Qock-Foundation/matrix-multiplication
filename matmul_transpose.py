import sys
import torch
from torch import nn
from utils import stringify_linear
from solve import solve


class TransposeMatmulModel(nn.Module):
    def __init__(self, n, m, r):
        super().__init__()
        self.n, self.m, self.r = n, m, r
        self.layer1 = nn.Linear(n * m, 2 * r, bias=False)
        self.layer2 = nn.Linear(r, n * n, bias=False)

    def __str__(self):
        result = ''
        syms = [f'A_{i // self.m}{i % self.m}' for i in range(self.n * self.m)]
        for t in range(2 * self.r):
            result += f'y_{t} = {stringify_linear(self.layer1.weight[t], syms)}\n'
        for t in range(self.r):
            result += f'z_{t} = y_{t} * y_{t + self.r}\n'
        syms = [f'z_{t}' for t in range(self.r)]
        for i in range(self.n * self.n):
            result += f'C_{i // self.n}{i % self.n} = {stringify_linear(self.layer2.weight[i], syms)}\n'
        return result

    def forward(self, x):
        x = self.layer1(x)
        x = x[:, :self.r] * x[:, self.r:]
        x = self.layer2(x)
        return x

    def sample(self, batch_size, scale):
        a = scale * torch.randn(batch_size, self.n, self.m)
        x = a.reshape(batch_size, -1)
        y = (a @ a.transpose(1, 2)).reshape(batch_size, -1)
        return x, y


n = int(sys.argv[1])
m = int(sys.argv[2])
r = int(sys.argv[3])
solve(lambda: TransposeMatmulModel(n, m, r))
