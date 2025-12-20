from typing import TypeAlias
from math import log2

import numpy as np

dxtype: TypeAlias = np.dtype
class DXTypes:
    @classmethod
    def __getattr__(cls, name:str) -> dxtype:
        if "x" in name:
            base, mult = name.split("x")[0:2]
            assert base[0] == "u"
            base = int(base[1:], base=10)
            mult = int(mult, base=10)
        else:
            base = name
            assert base[0] == "u"
            base = int(base[1:], base=10)
            mult = 1

        assert log2(base).is_integer()
        assert log2(mult).is_integer()
        if base > 64:
            words = base // 64
            base = 64
            dtype = np.dtype({
                'names': [str(i) for i in range(words)],
                'formats': [np.dtype(f"uint{base}")] * words,
                'offsets': [i*base for i in range(words)]
            })
        else:
            dtype = np.dtype(f"uint{base}")
        dtype = dxtype({
            'names': [str(i) for i in range(mult)],
            'formats': [dtype] * mult,
            'offsets': [i*base for i in range(mult)]
        })
        return dtype

DXT = DXTypes()


#print(DXT.u128x4)
print(np.zeros((8,), dtype=DXT.u512))
