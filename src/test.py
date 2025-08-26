import numpy as np

from neurray import *

masks = np.array([
0b00000001,
0b00001001]
, dtype=np.uint8)

vals = np.array([
0b00000001,
0b00001001]
, dtype=np.uint8)

test = np.array([8], dtype=np.uint8)


# o7 Elivrge
eliv = U1XToU1X(8, 8)
eliv.init_array()
eliv.set_training()
res = eliv.forward(masks)
print(res, vals)
eliv.backward(vals)
res = eliv.forward(masks)
print(res, vals)

res = eliv.forward(test)
print(res, test)


print(eliv.match, eliv.emit)

