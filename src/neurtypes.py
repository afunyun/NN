from typing import Annotated, Literal, TypeAlias
from math import log2
from abc import ABC, abstractmethod

import numpy as np
from numpy import _typing as np_ty
from numpy._core.numerictypes import uint64 # import error if np._typing


uint: TypeAlias = np.unsignedinteger[np_ty._8Bit | np_ty._16Bit | np_ty._32Bit | np_ty._64Bit]
luint: TypeAlias = list[np.dtype[np.uint64]]
dxtype: TypeAlias = np.dtype[uint] | Annotated[np.dtype[np.void], luint | list[np.dtype[uint] | luint]]

class _DXTypes:
    @staticmethod
    def __getattr__(name:str) -> dxtype:
        if "x" in name:
            base, mult = name.split("x")[0:2]
            assert base[0] == "u"
            base = int(base[1:], base=10)
            mult = int(mult, base=10)
        else:
            base = name
            assert base[0] == "u"
            assert len(base) > 2
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

# uhh so I'm not going to bother with the issue of u64+x1 being the same as u64x1+, I genually think I don't need to handle it. If anything, I'll take the u64+ route because that is more useful
def dxtype_args(dtype: dxtype) -> tuple[int, int]:
    if np.issubdtype(dtype, np.unsignedinteger):
        # <= u64 | mult == 1
        return dtype.itemsize*8, 1
    else:
        sub_dtype: uint | Annotated[np.void, luint] = dtype.fields["0"][0]
        if np.issubdtype(sub_dtype, np.unsignedinteger):
            # > u64 | mult == 1
            return sub_dtype.itemsize*8 * len(dtype.fields), 1
        else:
            # > u64 | mult > 1
            return len(sub_dtype.fields)*64, len(dtype.fields)


DXT = _DXTypes()



Neurray: TypeAlias = np.ndarray[tuple[Annotated[Literal[2], "Mirror"], Annotated[int, "Nodes"], Annotated[Literal[1], "Batch Size"]], dxtype]

State: TypeAlias = np.ndarray[tuple[Annotated[int, "Nodes"], Annotated[Literal[1], "Batch"]], dxtype]

Tokens: TypeAlias = np.ndarray[tuple[Annotated[int, "Batch"]], dxtype]


def tokens_to_neurray(tokens:Tokens) -> Neurray:
    state = np.reshape(tokens, (1, *tokens.shape))
    inverse = np.bitwise_not(state)
    output: Neurray = np.stack((state, inverse), axis=0) # pyright: ignore[reportAssignmentType]
    return output


class MatchBase(ABC):
    def __init__(self, nodes:int, match_size:dxtype, emit_size:dxtype):
        # 3d arrays :despair:
        self.match_inner = np.empty((2, nodes), dtype=match_size)
        self.match: Neurray = self.match_inner.reshape((*self.match_inner.shape, 1))
        assert self.match.base is self.match_inner
        self.emit: State = np.empty((nodes, 1), dtype=emit_size)

    # TODO need to properly do this in __init__ as well
    def post_init(self, match:Neurray, emit:State):
        assert dxtype_args(match.dtype) == dxtype_args(self.match.dtype)
        assert match.shape[0] == self.match.shape[0]
        assert dxtype_args(emit.dtype) == dxtype_args(self.emit.dtype)
        assert emit.shape[0] == self.emit.shape[0]
        self.match = match
        self.emit = emit

    @abstractmethod
    def forward(self, input_state:Tokens) -> Tokens: ...
    @abstractmethod
    def reverse(self, output_state:Tokens) -> tuple[Tokens, Tokens]: ...
    @abstractmethod
    def backwards(self, result:Tokens, expected_state:Tokens) -> Tokens: ...
