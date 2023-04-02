import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn

device = 'cuda'

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

def attempt(model, optimizer, criterion, num_epochs, batch_size, tol, alpha):
  model.train()
  for epoch in range(1, num_epochs + 1):
    for i in range(100):
      a = 3 * torch.randn(batch_size, n, n).to(device)
      b = 3 * torch.randn(batch_size, n, n).to(device)
      x = torch.cat((a.reshape(batch_size, n * n), b.reshape(batch_size, n * n)), 1)
      y = (a @ b).reshape(batch_size, n * n)
      z = model(x)
      loss = criterion(y, z)
      for p in model.parameters():
        loss += alpha * ((p - torch.round(p)) ** 2).sum()
      print('loss', loss)
      if loss < tol:
        break
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

def output_algorithm(model):
  with torch.no_grad():
    C1, C2 = model.parameters()
    for t in range(2 * k):
      print(f'y_{t} = ', end='')
      for ind in range(n ** 2):
        print(f'{+ round(C1[t][ind].item(), 3)} * A_{ind // n}{ind % n}', end='')
      for ind in range(n ** 2):
        print(f'{+ round(C1[t][n ** 2 + ind].item(), 3)} * B_{ind // n}{ind % n}', end='')
    for t in range(k):
      print(f'z_{t} = y_{t} * y_{t + k}')
    for ind in range(n ** 2):
      print(f'C_{ind // n}{ind % n} = ', end='')
      for t in range(k):
        print(f'+ {round(C2[ind][t].item(), 3)} * z_{t}', end='')

n, k = 3, 23
while True:
  model = MatmulModel(n, k).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  attempt(model, optimizer, nn.MSELoss(), num_epochs=20000, batch_size=1024, tol=1e-4, alpha=1e-4)
