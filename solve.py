import torch
import signal
from copy import deepcopy
from itertools import product
from tqdm import tqdm
from multiprocessing import Process, Queue, RLock
from time import time


def loss_function(x, y):
    return torch.sqrt(torch.mean((x - y) ** 2))


def custom_round(x, d):
    return torch.round(x * d) / d


def attempt(model, fixed, optimizer, scheduler, num_batches, batch_size, scale, tol, lock=RLock(), pid=0):
    signal.signal(signal.SIGTERM, lambda *args: exit(0))
    model.train()
    with lock:
        pbar = tqdm(total=num_batches, unit='batches', position=pid)
    for _ in range(num_batches):
        x, y = model.sample(batch_size, scale)
        z = model(x)
        loss = loss_function(y, z)
        mx = max([torch.max(torch.abs(arr)) for arr in model.parameters()])
        with lock:
            pbar.set_postfix(mx=f'{mx:.2f}', loss=f'{loss.item():.6f}', refresh=False)
            pbar.update()
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
    return False


def attempt_callback(model_class, model_args, fixed, lr, lr_decay, weight_decay,
                     num_batches, batch_size, scale, tol, lock, pid, seed, q):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = model_class(*model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=lr_decay,
                                                  total_iters=num_batches)
    success = attempt(model, fixed, optimizer, scheduler, num_batches, batch_size, scale, tol, lock, pid)
    q.put(model if success else None)
    signal.pause()


def solve(model_class,
          model_args,
          num_processes=4,
          num_batches1=400000,
          num_batches2=400000,
          num_batches3=400000,
          batch_size1=1024,
          batch_size2=256,
          batch_size3=1024,
          scale=1,
          weight_decay=1e-3,
          lr_decay=1e-3,
          lr1=1e-3,
          lr2=2e-4,
          lr3=1e-7,
          tol=1e-2,
          tol3=2e-6,
          denominators=(1, 2, 3, 4, 5, 6, 8, 9, 10, 12)):
    print('Approximation stage')
    fixed = []
    for arr in model_class(*model_args).parameters():
        fixed.append(torch.full_like(arr, torch.nan))
    model = None
    seed = time()
    while not model:
        lock = RLock()
        results = Queue()
        processes = []
        for pid in range(num_processes):
            process = Process(target=attempt_callback,
                              args=(model_class, model_args, fixed, lr1, lr_decay, weight_decay,
                                    num_batches1, batch_size1, scale, tol, lock, pid, seed, results))
            process.start()
            processes.append(process)
            seed += 1
        for _ in range(num_processes):
            model = results.get()
            if model:
                break
        for process in processes:
            process.terminate()
    print(model)

    print('Separation stage')
    indices = []
    for i, arr in enumerate(model.parameters()):
        prod = list(product(range(arr.shape[0]), range(arr.shape[1])))
        indices += zip([i] * len(prod), prod)
    for d in denominators:
        used = list(map(lambda arr: torch.zeros_like(arr, dtype=torch.bool), fixed[:-1]))
        while True:
            params = list(model.parameters())
            params_flat = torch.cat(list(map(torch.flatten, params[:-1])))
            fixed_flat = torch.cat(list(map(torch.flatten, fixed[:-1])))
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
                optimizer = torch.optim.Adam(model_new.parameters(), lr=lr2, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=lr_decay,
                                                              total_iters=num_batches2)
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
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=lr_decay,
                                                  total_iters=num_batches3)
    attempt(model, fixed, optimizer, scheduler,
            num_batches=num_batches3, batch_size=batch_size3, scale=scale, tol=tol3)
    print(model)
