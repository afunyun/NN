import numpy as np

from neurrayv3 import *
from neurtypes import DXT


def itr(neurray: U1XToU1X, tok: Tokens):
    res = neurray.forward(tok)
    rev = neurray.reverse(tok, res)
    opt, vals = neurray.assign(rev)
    neurray.apply(opt, vals)

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

test: Tokens = np.array([16], dtype=DXT.u8)
test2: Tokens = np.array([4], dtype=DXT.u8)

# o7 Elivrge

eliv = U1XToU1X(16, DXT.u8, DXT.u8)

itr(eliv, masks)

#res = eliv.forward(masks)
#assert res == vals
res = eliv.forward(masks)
#print(eliv.reverse(masks, res))

print("\nsep\n")
res = eliv.forward(test)
rev = eliv.reverse(test, res)
opt, vals = eliv.assign(rev)
#print(masks, res, rev, opt)
eliv.apply(opt, vals)


print("\nsep\n")
res = eliv.forward(test2)
rev = eliv.reverse(test2, res)
opt, vals = eliv.assign(rev)
#print(masks, res, rev, opt)
eliv.apply(opt, vals)


print(eliv.match_inner, eliv.emit_inner)
