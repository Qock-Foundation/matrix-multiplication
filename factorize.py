import sys
import torch
from descent import approximate, rationalize


def print_matrix(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            print('{:0.5f}'.format(A[i][j]), end=' ')
        print()


def build_matmul_tensor(n, m, p):
    A = torch.zeros((n * m, m * p, n * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A[m * i + j, p * j + k, i * p + k] = 1
    return A


def khatri_rao(U, V):
    return torch.einsum('ia,ja->ija', U, V).reshape((-1, U.shape[1]))


def last_factor(A3, U, V, eps=0.0):
    rank = U.shape[1]
    UV = khatri_rao(U, V)
    return (torch.linalg.inv(UV.T @ UV + eps / 2 * torch.eye(rank)) @ UV.T @ A3).T


def get_loss(A3, U, V, eps):
    W = last_factor(A3, U, V, eps)
    UV = khatri_rao(U, V)
    return torch.linalg.norm(UV @ W.T - A3)


if __name__ == '__main__':
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    p = int(sys.argv[3])
    r = int(sys.argv[4])
    A = build_matmul_tensor(n, m, p)
    A3 = A.reshape((-1, A.shape[2]))
    factor_shapes = [(A.shape[0], r), (A.shape[1], r)]
    factors = approximate(factor_shapes,
                          lambda factors: get_loss(A3, factors[0], factors[1], 1e-3),
                          num_iter=10000,
                          lr=0.2,
                          lr_decay=0.99,
                          eps=1e-3,
                          tol=1e-2,
                          debug=True)
    success = rationalize(factors,
                          lambda factors: get_loss(A3, factors[0], factors[1], 1e-3),
                          num_iter=5000,
                          lr=0.1,
                          lr_decay=0.95,
                          eps=1e-3,
                          tol=1e-2,
                          denominators=[0, 1, 2, 3, 4, 5, 6, 8],
                          debug=True)
    U, V = factors
    W = last_factor(A3, U, V)
    print_matrix(U)
    print()
    print_matrix(V)
    print()
    print_matrix(W)
    if not success:
        print('Failed to rationalize :(', file=sys.stderr)
        exit(1)
