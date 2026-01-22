import numpy as np

from neurrayv3 import *
from neurtypes import DXT

masks: Tokens = np.array([
0b00001001,
0b00000101,
0b10000000]
, dtype=DXT.u8)

"""
vals: Tokens = np.array([
0b00001001,
0b00000101]
, dtype=DXT.u8)
"""
test: Tokens = np.array([8], dtype=DXT.u8)


# o7 Elivrge

eliv = U1XToU1X(16, DXT.u8, DXT.u8)
#eliv = UXToUX(8,8,4)
res = eliv.forward(masks)
rev = eliv.reverse(masks, res)
opt, vals = eliv.assign(rev)
#print(masks, res, rev, opt)
eliv.apply(rev, vals)
res = eliv.forward(masks)
#assert res == vals

res = eliv.forward(test)
print(test, res)


print(eliv.match_inner, eliv.emit_inner)
