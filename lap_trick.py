"""
    lap_trick.py
    
    This is a simple trick, but has given us large speedups in practice:
        
        If C is a r x c matrix, then we know that the column that is ultimately
        chosen to match row i must be among the r smallest entries in that row.
        
        Thus, we can delete columns of C that do not contain one of the r smallest 
        entries for any row.
        
        This implementation illustrates the idea - it could go faster w/ parallelization,
        etc.
"""

import numpy as np
from time import time
from scipy.optimize import linear_sum_assignment

r = 5
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
topk      = np.argpartition(C, r, axis=-1)[:,:r] # !! This takes all of the time ... but can be parallelized
topk      = np.unique(topk)
_, c_idx2 = linear_sum_assignment(C[:,topk])
c_idx2    = topk[c_idx2]
elapsed   = time() - t
print(f'trick | elapsed={elapsed}')

# --
# Check

assert C[(np.arange(r), c_idx)].sum() == C[(np.arange(r), c_idx2)].sum()
