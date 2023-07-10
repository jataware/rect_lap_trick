import numpy as np
from time import time
from scipy.optimize import linear_sum_assignment

r = 20
c = 10_000_000
C = np.random.choice(range(256), (r, c))

# --
# Normal

t        = time()
_, c_idx = linear_sum_assignment(C)
elapsed  = time() - t
print(f'orig  | elapsed={elapsed}')

# --
# Trick

t         = time()
topk      = np.unique(np.argpartition(C, r, axis=-1)[:,:r])
_, c_idx2 = linear_sum_assignment(C[:,topk])
c_idx2    = topk[c_idx2]
elapsed   = time() - t
print(f'trick | elapsed={elapsed}')

# --
# Check

assert C[(np.arange(r), c_idx)].sum() == C[(np.arange(r), c_idx2)].sum()