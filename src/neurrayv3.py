import numpy as np
from numpy.ma.core import swapaxes

from neurtypes import MatchBase, Neurray, Tokens, tokens_to_neurray

class U1XToU1X(MatchBase):
    def forward(self, input_state: Tokens) -> Tokens:
        assert input_state.dtype == self.match.dtype
        input: Neurray = tokens_to_neurray(input_state)
        choices = np.bitwise_or(input, self.match).all(axis=0)
        broad = np.broadcast_to(self.emit, (self.count, input.shape[2]))
        output: Tokens = np.bitwise_or.reduce(broad, axis=-2, where=choices)
        return output

    # welcome back UXtoUX
    def reverse(self, input_state:Tokens, output_state: Tokens) -> np.ndarray:
        assert output_state.dtype == self.emit.dtype
        assert input_state.shape[0] == output_state.shape[0]
        reps = np.broadcast_to(self.match, (2, self.count, output_state.shape[0]))
        mask2 = np.broadcast_to((np.bitwise_and(output_state, self.emit) != 0 ), (2, self.count, output_state.shape[0]))
        red = np.bitwise_or.reduce(reps, axis=1, where=mask2 )
        diff = np.bitwise_xor(red, np.stack((input_state, np.bitwise_invert(input_state))))
        # removing [0, 0] instances, because useless to compute
        diff_cul = diff[diff.any(axis=-1)]
        return diff_cul

    def assign(self, diff: np.ndarray) -> tuple[np.ndarray, Tokens]:
        """HELP MEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"""
        # Batch normalization :RAGEY:

        def reduce_duplicates(arr:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # such a weird thing, it works at least; make sure to match dtype or it'll promote
            aran = np.arange(arr.shape[-2], dtype=arr.dtype)
            agrid = (aran[..., None] + aran) % arr.shape[-2]
            arr2 = arr[agrid]
            # check if anything matches the first row
            mask = np.triu(arr2[..., 0, 0] == arr2[... , 0])
            # mask out if a row matches so we don't do it twice
            mask2 = np.bitwise_invert(mask[1:].any(axis=0))
            # reduce the negitive of positive matches
            arr_neg_cul = np.bitwise_and.reduce(arr2[..., 1], axis=0, where=mask)
            arr_cul = np.stack((arr[mask2, 0], arr_neg_cul[mask2]))
            return arr_cul, mask2

        diff2 = (diff[..., None] & self.match_inner[:, None, :])
        mask2 = np.bitwise_invert(diff2[0].any(axis=1))
        mask3 = diff2.all(axis=0)
        # botch swap axes call, might do that later once all the rest of the pain is dealt with
        diff3 = np.swapaxes(np.concat((diff[..., mask2], diff2[..., diff2[0]!=0]), axis=1), -1, -2)
        diff2_cul, active = reduce_duplicates(diff3)

        # the diff has now been seperated into components, next is proceeding to generate the new branches
        assert self.size + diff2_cul.shape[0] <= self.limit, "out of states"
        shift = np.ones(diff2_cul.shape[0] + 1, dtype=self.emit_inner.dtype) << np.arange(self.used, self.used + diff2_cul.shape[0] + 1, dtype=self.emit_inner.dtype) # pyright: ignore[reportOperatorIssue]
        self.size += diff2_cul.shape[0]
        temp_name = np.broadcast_to(self.emit_inner,(diff2_cul.shape[1], self.emit_inner.shape[0]))
        res = np.bitwise_or.reduce(temp_name, axis=1, where=mask3[active]) ^ shift

        return diff2_cul, res

    def apply(self, match: np.ndarray, emit: Tokens) -> None:
        assert match.shape[1] == emit.shape[0]
        assert match.shape[0] == 2

        # (1, X)
        reshaped_emit = np.broadcast_to(emit[..., None], (*emit.shape, self.count))
        reshaped_match = np.broadcast_to(match[..., None], (*match.shape, self.count))

        # EXACT
        # (1, X) and (Y, 1)
        hard_mask = np.equal(self.emit_inner[None, ...], reshaped_emit)
        hard_apply_mask = hard_mask.any(axis=0)
        hard_used = hard_mask.any(axis=1)

        hard_emit_apply = np.bitwise_and.reduce(reshaped_emit, axis=0, where=hard_mask)
        self.match_inner[..., hard_apply_mask] = hard_emit_apply[..., hard_apply_mask]

        # CONTAINS
        loose_mask = np.bitwise_and(self.emit_inner[None, ...], reshaped_emit) != 0
        loose_emit_apply = np.bitwise_and.reduce(reshaped_emit, axis=0, where=loose_mask)
        loose_match_apply = np.bitwise_and.reduce(reshaped_match, axis=1, where=loose_mask)
        loose_apply_mask = loose_mask.any(axis=0)
        loose_used = loose_mask.any(axis=1)

        loose_new_match = self.match_inner[..., loose_apply_mask] ^ loose_match_apply[..., loose_apply_mask]
        loose_new_emit = self.emit_inner[loose_apply_mask] ^ loose_emit_apply[loose_apply_mask]

        loose_count = loose_new_match.shape[1]

        self.match_inner[..., loose_apply_mask] &= loose_match_apply[:, loose_apply_mask]
        self.emit_inner[loose_apply_mask] &= loose_emit_apply[loose_apply_mask]

        assert loose_count + self.used <= self.count, "out of slots"
        self.match_inner[...,self.used:self.used + loose_count] = loose_new_match
        self.emit_inner[..., self.used:self.used + loose_count] = loose_new_emit
        self.used += loose_count

        # UNDISCOVERD
        empty_mask = np.bitwise_invert(np.bitwise_or(hard_used, loose_used))
        new_match = match[..., empty_mask]
        new_emit = emit[empty_mask]

        new_count = new_emit.shape[0]

        assert new_count + self.used <= self.count, "out of slots"
        self.match_inner[...,self.used:self.used + new_count] = new_match
        self.emit_inner[..., self.used:self.used + new_count] = new_emit
        self.used += new_count
