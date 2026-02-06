import numpy as np

from neurrayv4 import U1XToU1X

eliv = U1XToU1X(np.empty(4, dtype=np.uint8), np.uint8, cases=6)

res, rev = eliv.forward(np.array([[8, 0, 0, 0], [2,0,0,0], [4,0,0,0]], dtype=np.uint8))
eliv.assign(rev)

res, rev = eliv.forward(np.array([[9, 0, 0, 0], [12,0,0,0], [14,0,0,0]], dtype=np.uint8))
eliv.assign(rev)

res, rev = eliv.forward(np.array([[9, 0, 0, 0], [12,0,0,0], [14,0,0,0]], dtype=np.uint8))
print(res)

print(eliv.match, eliv.array_used)
