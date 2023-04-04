import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn
from joblib import Parallel, delayed

device = 'cpu'
m, n, p, k = 2, 2, 5, 17

class MatmulFedroModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer11 = nn.Linear(m * n, k, bias=False)
    self.layer12 = nn.Linear(n * p, k, bias=False)
    self.layer3 = nn.Linear(k, m * p, bias=False)
  def forward(self, a, b):
    a, b = a.reshape(-1, m * n), b.reshape(-1, n * p)
    return self.layer3(self.layer11(a) * self.layer12(b))

def attempt(fixed):
  num_iters, batch_size, tol, alpha = 100000, 1024, 1e-3, 0
  model = MatmulFedroModel().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1-5/num_iters))
  criterion = nn.MSELoss()
  model.train()
  for iteration in range(0, num_iters + 1):
    if iteration % 30 == 0:
      frac_thr = iteration / (2 * num_iters)
      print('frac_thr', frac_thr)
      all_params = torch.cat([param.ravel() for param in model.parameters()]).cpu().detach().numpy()
      fixed = np.full((m * n + n * p + p * m) * k, 57)
      mask = np.abs(all_params - np.round(all_params)) < frac_thr
      fixed[mask] = (all_params[mask] + 0.5).astype(int)
      with torch.no_grad():
        for param in model.parameters():
          mask = torch.abs(param - torch.round(param)) < frac_thr
          param[mask] = torch.round(param[mask])
      m11 = fixed[:m * n * k].reshape(k, m * n)
      m12 = fixed[m * n * k:(m * n + n * p) * k].reshape(k, n * p)
      m3 = fixed[(m * n + n * p) * k:].reshape(m * p, k)
      m11_mask = torch.BoolTensor(m11 != 57).to(device)
      m12_mask = torch.BoolTensor(m12 != 57).to(device)
      m3_mask = torch.BoolTensor(m3 != 57).to(device)
    a = torch.randn(batch_size, m, n).to(device)
    b = torch.randn(batch_size, n, p).to(device)
    y = (a @ b).reshape(batch_size, m * p)
    z = model(a, b)
    loss = criterion(y, z)
    if iteration % 100 == 0:
      print('loss', loss, 'params', torch.cat([param.ravel() for param in model.parameters()]).cpu().detach().tolist()[:8])
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

def f_frac(fixed):   # maximize
  return np.mean(Parallel(n_jobs=24)(delayed(attempt)(fixed) for i in range(24)))
def f_nones(fixed):  # minimize
  return np.sum(fixed == 57)

fixed = np.full((m * n + n * p + p * m) * k, 57)
attempt(fixed)
quit(0)

#assert f(np.random.randint(3, size=(m*n+n*p)*(2*k)+(m*p)*k) - 1) == 0
#assert f(np.full(fill_value=57, shape=(m*n+n*p)*(2*k)+(m*p)*k)) == 1

fixed = np.full((m * n + n * p + p * m) * k, 57)
assert f_frac(fixed) > 0.2
for Temp in np.exp(np.linspace(np.log(0.1), np.log(0.001), 3000)):
  i_ch = np.random.randint((m * n + n * p + p * m) * k)
  old_v = fixed[i_ch]
  v_frac, v_nones = f_frac(fixed), f_nones(fixed)
  old_value = v_frac - 0.01 * v_nones
  fixed[i_ch] = np.random.choice([57, 0, -1, +1])
  v_frac, v_nones = f_frac(fixed), f_nones(fixed)
  new_value = v_frac - 0.01 * v_nones
  accept = new_value > old_value or np.random.random() < np.exp((new_value - old_value) / Temp)
  print(fixed, old_value, new_value, 'frac', v_frac, 'nones', v_nones, 'accept' if accept else 'reject')
  if not accept:
    fixed[i_ch] = old_v
print('fixed:', fixed, 'frac', f_frac(fixed), 'nones', f_nones(fixed))
