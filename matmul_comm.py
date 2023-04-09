import sys
from models import CommMatmulModel, solve


def model_creator():
    return CommMatmulModel(n, m, p, r)


n = int(sys.argv[1])
m = int(sys.argv[2])
p = int(sys.argv[3])
r = int(sys.argv[4])
solve(model_creator)
