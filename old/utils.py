import torch


def stringify_monom(c, sym, eps):
    if abs(c) < eps:
        return '0'
    if abs(c - 1) < eps:
        return sym
    if abs(c + 1) < eps:
        return '-' + sym
    return str(round(c, 5)) + ' * ' + sym


def stringify_linear(cs, syms, eps=1e-2):
    nonzero = torch.nonzero(torch.abs(cs) >= eps)
    return ' + '.join([stringify_monom(cs[i].item(), syms[i], eps) for i in nonzero])
