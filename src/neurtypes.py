from typing import Annotated, Any, Literal, TypeAlias
from math import log2

import numpy as np
from numpy import _typing as np_ty



uint: TypeAlias = np.dtype[np.unsignedinteger[np_ty._8Bit | np_ty._16Bit | np_ty._32Bit | np_ty._64Bit]]
dxtype: TypeAlias = uint | Annotated[np.dtype[np.void], list[np.uint64] | list[uint | list[np.uint64]]]

class _DXTypes:
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
        if mult > 1:
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
            dtype = np.dtype({
                'names': [str(i) for i in range(mult)],
                'formats': [dtype] * mult,
                'offsets': [i*base for i in range(mult)]
            })
            return dtype
        else:
            if base > 64:
                words = base // 64
                base = 64
                dtype = np.dtype({
                    'names': [str(i) for i in range(words)],
                    'formats': [np.dtype(f"uint{base}")] * words,
                    'offsets': [i*base for i in range(words)]
                })
                return dtype
            else:
                dtype = np.dtype(f"uint{base}")
                return dtype


DXT = _DXTypes()



Neurray: TypeAlias = np.ndarray[tuple[int, Annotated[int,Literal[2]], Annotated[int,Literal[1]]], dxtype]
def create_neurray(paths:int, dtype: dxtype) -> Neurray:
    return np.empty((paths, 2, 1), dtype=dtype)



class MatchBase:
    def __init__(self, nodes:int, match_size:dxtype, emit_size:dxtype):
        # 3d arrays :despair:
        self.match: Neurray = create_neurray(nodes, match_size)
        self.emit: Neurray = create_neurray(nodes, emit_size)
