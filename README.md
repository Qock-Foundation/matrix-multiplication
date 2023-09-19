# matrix-multiplication

This repo contains python scripts for generating fast algorithms for matrix multiplication of various sizes.

The input matrices have dimensions `(n, m)`, `(m, p)`, and `r` is the desired number of basic multiplications.

Classic (non-commutative case):
```
python3 matmul_classic <n> <m> <p> <r>
```

Commutative case:
```
python3 matmul_commutative <n> <m> <p> <r>
```

In the classical case, for `n = m = p = 2` and `r = 7` it takes less than 30 seconds to find the Strassen's algorithm.
