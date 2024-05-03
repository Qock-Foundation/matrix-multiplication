import sys
import torch
import numpy as np
from tqdm import tqdm
from itertools import product
from copy import deepcopy


def custom_round(x, d):
    return torch.round(x * d) / d if d != 0 else torch.zeros_like(x)


def sorted_order(tensors):
    nums, locs = [], []
    for i, tensor in enumerate(tensors):
        locs += list(product([range(shape) for shape in tensor.shape]))
        nums += [i] * len(locs)
    tensors_flat = torch.cat(list(map(torch.flatten, tensors)))
    order = torch.argsort(tensors_flat)
    return [(nums[i], locs[i]) for i in order]


def attempt(params, fixed, loss_func, num_iter, lr, lr_decay, eps, tol, debug):
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay)
    pbar = tqdm(total=num_iter, unit='batches', leave=False) if debug else None
    for _ in range(num_iter):
        loss = loss_func(params)
        mx = torch.max(torch.abs(torch.cat(list(map(torch.flatten, params)))))
        if debug:
            pbar.set_postfix(loss=f'{loss.item():.6f}', mx=f'{mx:.2f}')
            pbar.update()
        if loss < tol:
            return True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        with torch.no_grad():
            for p, f in zip(params, fixed):
                p[~f.isnan()] = f[~f.isnan()]
    return False


def try_round(params, target, fixed, used, loss_func, num_iter, lr, lr_decay, eps, tol, debug):
    min_delta = torch.inf
    min_ind = None
    min_param_id = None
    for param_id in range(len(params)):
        deltas = torch.abs(params[param_id] - target[param_id])
        deltas[~fixed[param_id].isnan() | used[param_id]] = torch.inf
        delta = torch.min(deltas)
        ind = torch.argmin(deltas)
        if delta < min_delta:
            min_delta = delta
            min_ind = ind
            min_param_id = param_id
    min_indices = np.unravel_index(min_ind, params[min_param_id].shape)
    used[min_param_id][min_indices] = True
    cnt_params = sum([np.prod(p.shape) for p in params])
    cnt_fixed = sum([(~f.isnan()).sum().item() for f in fixed])
    if debug:
        print(f'[{cnt_fixed + 1}/{cnt_params}] separating '
              f'{params[min_param_id][min_indices]} -> {target[min_param_id][min_indices]}       ',
              file=sys.stderr)
    params_new = deepcopy(params)
    fixed_new = deepcopy(fixed)
    with torch.no_grad():
        params_new[min_param_id][min_indices] = target[min_param_id][min_indices]
        fixed_new[min_param_id][min_indices] = target[min_param_id][min_indices]
    success = attempt(params_new, fixed_new, loss_func, num_iter, lr, lr_decay, eps, tol, debug)
    if debug:
        print('\033[F', end='', file=sys.stderr)
        sys.stderr.flush()
    if success:
        with torch.no_grad():
            for param_id in range(len(params)):
                params[param_id] = params_new[param_id]
                fixed[param_id] = fixed_new[param_id]
        return True
    return False


def approximate(param_shapes, loss_func, num_iter, lr, lr_decay, eps, tol, debug=False):
    while True:
        params = [torch.randn(shape, requires_grad=True) for shape in param_shapes]
        fixed = [torch.full(shape, torch.nan) for shape in param_shapes]
        success = attempt(params, fixed, loss_func, num_iter, lr, lr_decay, eps, tol, debug)
        if success:
            return params


def rationalize(params, loss_func, num_iter, lr, lr_decay, eps, tol, denominators, debug=False):
    fixed = [torch.full_like(p, torch.nan) for p in params]
    for d in denominators:
        used = [torch.zeros_like(p, dtype=torch.bool) for p in params]
        while any([(fixed[param_id].isnan() & ~used[param_id]).any() for param_id in range(len(params))]):
            target = [custom_round(p, d) for p in params]
            try_round(params, target, fixed, used, loss_func, num_iter, lr, lr_decay, eps, tol, debug)
    if any([f.isnan().any() for f in fixed]):
        return False
    return True
