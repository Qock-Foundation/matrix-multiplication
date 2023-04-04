import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn
from joblib import Parallel, delayed

device = 'cpu'
m, n, p, k = 2, 2, 3, 10

class MatmulFedroModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer11 = nn.Linear(m * n, k, bias=False)
    self.layer12 = nn.Linear(n * p, k, bias=False)
    self.layer3 = nn.Linear(k, m * p, bias=False)
  def forward(self, a, b):
    a, b = a.reshape(-1, m * n), b.reshape(-1, n * p)
    return self.layer3(self.layer11(a) * self.layer12(b))

def attempt(fixed, num_iters=10000, batch_size=1024, tol=1e-4, alpha=0):
  m11 = fixed[:m * n * k].reshape(k, m * n)
  m12 = fixed[m * n * k:(m * n + n * p) * k].reshape(k, n * p)
  m3 = fixed[(m * n + n * p) * k:].reshape(m * p, k)
  m11_mask = torch.BoolTensor(m11 != 57).to(device)
  m12_mask = torch.BoolTensor(m12 != 57).to(device)
  m3_mask = torch.BoolTensor(m3 != 57).to(device)
  model = MatmulFedroModel().to(device)
  with torch.no_grad():
    model.layer11.weight[m11_mask] = torch.FloatTensor(m11).to(device)[m11_mask]
    model.layer12.weight[m12_mask] = torch.FloatTensor(m12).to(device)[m12_mask]
    model.layer3.weight[m3_mask ] = torch.FloatTensor(m3).to(device)[m3_mask]
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1-5/num_iters))
  criterion = nn.MSELoss()
  model.train()
  for iteration in range(1, num_iters + 1):
    a = 3 * torch.randn(batch_size, m, n).to(device)
    b = 3 * torch.randn(batch_size, n, p).to(device)
    y = (a @ b).reshape(batch_size, m * p)
    z = model(a, b)
    loss = criterion(y, z)
    #for p in model.parameters():
    #  loss += alpha * ((p - torch.round(p)) ** 2).sum()
    if loss < tol:
      return True
    optimizer.zero_grad()
    loss.backward()
    model.layer11.weight.grad[m11_mask] = 0
    model.layer12.weight.grad[m12_mask] = 0
    model.layer3.weight.grad[m3_mask] = 0
    optimizer.step()
    scheduler.step()
  return False

def output_algorithm(model):
  with torch.no_grad():
    C1, C2 = model.parameters()
    for t in range(2 * k):
      print(f'y_{t} =', end='')
      for ind in range(n ** 2):
        print(f' + {round(C1[t][ind].item(), 3)} * A_{ind // n}{ind % n}', end='')
      for ind in range(n ** 2):
        print(f' + {round(C1[t][n ** 2 + ind].item(), 3)} * B_{ind // n}{ind % n}', end='')
    for t in range(k):
      print(f'z_{t} = y_{t} * y_{t + k}')
    for ind in range(n ** 2):
      print(f'C_{ind // n}{ind % n} =', end='')
      for t in range(k):
        print(f' + {round(C2[ind][t].item(), 3)} * z_{t}', end='')
    print()

def f_frac(fixed):   # maximize
  return np.mean(Parallel(n_jobs=24)(delayed(attempt)(fixed) for i in range(24)))
def f_nones(fixed):  # minimize
  return np.sum(fixed == 57)
def f(fixed):        # maximize
  return f_frac(fixed) - 0.01 * f_nones(fixed)

#assert f(np.random.randint(3, size=(m*n+n*p)*(2*k)+(m*p)*k) - 1) == 0
#assert f(np.full(fill_value=57, shape=(m*n+n*p)*(2*k)+(m*p)*k)) == 1

fixed = np.full((m * n + n * p + p * m) * k, 57)
for Temp in np.exp(np.linspace(np.log(1), np.log(0.001), 1000)):
  i_ch = np.random.randint((m * n + n * p + p * m) * k)
  old_v = fixed[i_ch]
  old_value = f(fixed)
  fixed[i_ch] = np.random.choice([57, 0, -1, +1])
  new_value = f(fixed)
  accept = new_value > old_value or np.random.random() < np.exp((new_value - old_value) / Temp)
  print(fixed, old_value, new_value, 'accept' if accept else 'reject')
  if not accept:
    fixed[i_ch] = old_v
print('fixed:', fixed, 'frac', f_frac(fixed), 'nones', f_nones(fixed))
