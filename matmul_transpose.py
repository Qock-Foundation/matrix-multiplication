import sys
from models import TransposeMatmulModel, solve


def model_creator():
    return TransposeMatmulModel(n, m, r)


n = int(sys.argv[1])
m = int(sys.argv[2])
r = int(sys.argv[3])
solve(model_creator)
