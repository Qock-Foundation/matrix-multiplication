import torch
from copy import deepcopy
from itertools import product
from tqdm import trange


def loss_function(x, y):
    return torch.sqrt(torch.mean((x - y) ** 2))


def custom_round(x, d):
    return torch.round(x * d) / d


def attempt(model, fixed, optimizer, scheduler, num_batches, batch_size, scale, tol):
    model.train()
    pbar = trange(num_batches, unit='batches')
    for _ in pbar:
        x, y = model.sample(batch_size, scale)
        z = model(x)
        loss = loss_function(y, z)
        pbar.set_postfix(loss=f'{loss.item():.6f}', refresh=False)
        if loss.isnan():
            return False
        if loss < tol:
            return True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for i, arr in enumerate(model.parameters()):
                mask = ~fixed[i].isnan()
                arr[mask] = fixed[i][mask]


def solve(model_creator,
          num_batches1=400000,
          num_batches2=300000,
          num_batches3=500000,
          batch_size1=1024,
          batch_size2=256,
          batch_size3=1024,
          gamma=1 - 5 / 400000,
          scale=2,
          lr1=1e-3,
          lr2=2e-4,
          lr3=1e-6,
          tol=1e-2,
          tol3=1e-6,
          denominators=(1, 2, 3, 4, 5, 6, 8, 9, 10, 12)):
    print('Approximation stage')
    fixed = []
    for params in model_creator().parameters():
        fixed.append(torch.full_like(params, torch.nan))
    while True:
        model = model_creator()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        if attempt(model, fixed, optimizer, scheduler,
                   num_batches=num_batches1, batch_size=batch_size1, scale=scale, tol=tol):
            break
    print(model)

    print('Separation stage')
    for d in denominators:
        used = list(map(lambda arr: torch.zeros_like(arr, dtype=torch.bool), fixed[:-1]))
        while True:
            params = list(model.parameters())
            params_flat = torch.cat(list(map(torch.flatten, params[:-1])))
            fixed_flat = torch.cat(list(map(torch.flatten, fixed[:-1])))
            indices = []
            for i, arr in enumerate(params[:-1]):
                prod = list(product(range(arr.shape[0]), range(arr.shape[1])))
                indices += zip([i] * len(prod), prod)
            dist = torch.abs(params_flat - custom_round(params_flat, d))
            num_params = len(params_flat)
            num_fixed = sum(~fixed_flat.isnan())
            for pos in torch.argsort(dist):
                i, loc = indices[pos]
                if used[i][loc] or not fixed[i][loc].isnan():
                    continue
                used[i][loc] = True
                old_value = params[i][loc]
                new_value = custom_round(old_value, d)
                print(f'[{num_fixed + 1}/{num_params}] separating {old_value} -> {new_value}')
                model_new = deepcopy(model)
                params_new = list(model_new.parameters())
                fixed_new = deepcopy(fixed)
                with torch.no_grad():
                    params_new[i][loc] = new_value
                    fixed_new[i][loc] = new_value
                optimizer = torch.optim.Adam(model_new.parameters(), lr=lr2)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
                if attempt(model_new, fixed_new, optimizer, scheduler,
                           num_batches=num_batches2, batch_size=batch_size2, scale=scale, tol=tol):
                    model = model_new
                    fixed = fixed_new
                    break
            else:
                break
    print(model)

    print('Refinement stage')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    attempt(model, fixed, optimizer, scheduler,
            num_batches=num_batches3, batch_size=batch_size3, scale=scale, tol=tol3)
    print(model)
