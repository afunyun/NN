from typing import Annotated, Self, TypeAlias, Literal

import numpy as np

class Rdu:
    __array_priority__ = 1000
    def __init__(self, axis:int|tuple[int]=-1):
        self.axis = axis
    def __rand__(self, left:np.ndarray) -> Self:
        self.l = left
        self.f = 0
        return self
    def __ror__(self, left:np.ndarray) -> Self:
        self.l = left
        self.f = 1
        return self

    def __eq__(self, right:np.ndarray) -> np.ndarray: # pyright: ignore[reportIncompatibleMethodOverride]
        if self.f:
            return np.bitwise_or.reduce(array=self.l, axis=self.axis, where=right)
        else:
            return np.bitwise_and.reduce(array=self.l, axis=self.axis, where=right)



def case_gen_64(dtype:np.dtype[np.unsignedinteger]|type[np.unsignedinteger]) -> np.ndarray:
    size = np.iinfo(dtype).bits

    arrs = [np.arange(0,n, dtype=dtype) for n in range(size-1, 0, -1)]
    arr2 = [np.array([n] * n, dtype=dtype) for n in range(size-1, 0, -1)]

    ones1 = np.ones(((size) * (size-1)) // 2, dtype=dtype)
    idx = np.concat(arrs)
    idx2 = np.concat(arr2)
    ones2 = np.copy(ones1)

    cases = ones1 << idx | ones2 << idx2

    cases_rolled_list = [cases[0]]
    i = size
    while i != 0:
        cases_rolled_list.append(cases[i])
        i = (i + size + 1) % ((size) * (size-1) // 2)
    cases_rolled = np.array(cases_rolled_list)
    return cases_rolled



Case: TypeAlias = np.ndarray[tuple[int], np.dtype[np.unsignedinteger]]

State: TypeAlias = Annotated[np.ndarray[tuple[int, int], np.dtype[np.unsignedinteger]],   tuple[Case]]

Diff: TypeAlias = np.ndarray[tuple[int, Literal[2], int], np.dtype[np.unsignedinteger]]
