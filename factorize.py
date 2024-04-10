import sys
import torch
from tqdm import tqdm
from itertools import product
from copy import deepcopy


def print_matrix(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            print('{:0.5f}'.format(A[i][j]), end=' ')
        print()


def custom_round(x, d):
    return torch.round(x * d) / d if d != 0 else torch.zeros_like(x)


def sorted_order(tensors):
    nums, locs = [], []
    for i, tensor in enumerate(tensors):
        locs += list(product(range(tensor.shape[0]), range(tensor.shape[1])))
        nums += [i] * len(locs)
    tensors_flat = torch.cat(list(map(torch.flatten, tensors)))
    order = torch.argsort(tensors_flat)
    return [(nums[i], locs[i]) for i in order]


def build_matmul_tensor(n, m, p):
    A = torch.zeros((n * m, m * p, n * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A[m * i + j, p * j + k, i * p + k] = 1
    return A


def khatri_rao(U, V):
    return torch.einsum('ia,ja->ija', U, V).reshape((-1, U.shape[1]))


def last_factor(A3, U, V, eps=0):
    rank = U.shape[1]
    UV = khatri_rao(U, V)
    return (torch.linalg.inv(UV.T @ UV + eps / 2 * torch.eye(rank)) @ UV.T @ A3).T


def get_loss(A3, U, V, W):
    UV = khatri_rao(U, V)
    return torch.linalg.norm(UV @ W.T - A3)


def attempt(A, params, fixed, num_iter, lr, lr_decay, eps, tol, debug):
    A3 = A.reshape((-1, A.shape[2]))
    optimizer = torch.optim.Adam([params], lr=lr, weight_decay=eps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay)
    pbar = tqdm(total=num_iter, unit='batches') if debug else None
    for _ in range(num_iter):
        U, V = torch.split(params, A.shape[:2])
        W = last_factor(A3, U, V, eps)
        loss = get_loss(A3, U, V, W)
        mx = torch.max(torch.abs(torch.cat((params, W))))
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
            params[~fixed.isnan()] = fixed[~fixed.isnan()]
    return False


def factorize(A, rank,
              num_iter=10000,
              lr=0.2,
              lr_decay=0.99,
              eps=1e-3,
              tol=1e-2,
              debug=False):
    A3 = A.reshape((-1, A.shape[2]))
    while True:
        params = torch.randn((sum(A.shape[:2]), rank), requires_grad=True)
        success = attempt(A, params, torch.full_like(params, torch.nan),
                          num_iter, lr, lr_decay, eps, tol, debug)
        if success:
            U, V = torch.split(params, A.shape[:2])
            W = last_factor(A3, U, V)
            return [U, V, W]


def try_round(params, target, fixed, used, num_iter, lr, lr_decay, eps, tol, debug):
    delta = torch.abs(params - target)
    delta[~fixed.isnan() | used] = torch.inf
    ind = torch.argmin(delta).item()
    i, j = divmod(ind, params.shape[1])
    used[i][j] = True
    cnt_params = params.shape[0] * params.shape[1]
    cnt_fixed = (~fixed.isnan()).sum().item()
    if debug:
        print(f'[{cnt_fixed + 1}/{cnt_params}] separating {params[i][j]} -> {target[i][j]}',
              file=sys.stderr)
    params_new = deepcopy(params)
    fixed_new = deepcopy(fixed)
    with torch.no_grad():
        params_new[i][j] = target[i][j]
        fixed_new[i][j] = target[i][j]
    success = attempt(A, params_new, fixed_new, num_iter, lr, lr_decay, eps, tol, debug)
    if success:
        with torch.no_grad():
            params[:] = params_new
            fixed[:] = fixed_new
        return True
    return False


def rationalize_factors(A, factors,
                        num_iter=5000,
                        lr=0.1,
                        lr_decay=0.95,
                        eps=1e-3,
                        tol=1e-2,
                        denominators=(0, 1, 2, 3, 4, 5, 6, 8),
                        debug=False):
    A3 = A.reshape((-1, A.shape[2]))
    params = torch.cat(factors[:2]).detach().requires_grad_(True)
    fixed = torch.full_like(params, torch.nan)

    for d in denominators:
        used = torch.zeros_like(params, dtype=torch.bool)
        while (fixed.isnan() & ~used).any():
            target = custom_round(params, d)
            try_round(params, target, fixed, used, num_iter, lr, lr_decay, eps, tol, debug)

    if fixed.isnan().any():
        return False
    factors[0], factors[1] = params.split(A.shape[:2])
    factors[2] = last_factor(A3, factors[0], factors[1])
    return True


if __name__ == '__main__':
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    p = int(sys.argv[3])
    r = int(sys.argv[4])
    A = build_matmul_tensor(n, m, p)
    factors = factorize(A, r, debug=True)
    success = rationalize_factors(A, factors, debug=True)
    U, V, W = factors
    print_matrix(U)
    print()
    print_matrix(V)
    print()
    print_matrix(W)
    if not success:
        print('Failed to rationalize :(', file=sys.stderr)
        exit(1)
