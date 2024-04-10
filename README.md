# matrix-multiplication

This repository contains python scripts for finding the CP-decomposition of a 3d tensor.
The methods are quite efficient for generating fast algorithms for matrix multiplication of small sizes.
Namely, it is possible to find the 3x3-matrix multiplication algorithm with 23 multiplications, which is extremely hard for other well-known approaches such as ALS.

The input matrices have dimensions `(n, m)`, `(m, p)`, and `r` is the desired number of basic multiplications (aka tensor rank).

```
python3 factorize.py n m p r
```

The directory `factors` contains already computed factorizations for various small sizes.