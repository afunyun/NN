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
        red = np.bitwise_or.reduce(reps, axis=1, where=~mask2)
        diff = np.bitwise_xor(red, np.stack((input_state, np.bitwise_invert(input_state))))
        # removing [0, 0] instances, because useless to compute
        diff_cul = diff[diff.any(axis=-1)]
        return diff_cul

    def assign(self, diff: np.ndarray) -> tuple[np.ndarray, Tokens]:
        """HELP MEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"""
        # Batch normalization :RAGEY:
        assert diff.shape[0] == 2, "diff is malformed"


        diff2 = np.bitwise_and(diff[..., None], self.match_inner[:, None, :])
        mask2 = diff2[0].all(axis=1)

        mask4 = diff2[:, mask2].all(axis=0)

        passed = diff[0, ~mask2]

        redu = np.bitwise_and.reduce(diff2[0, mask2], axis=1, where=mask4)
        diff3_pos = np.unique(np.concat((passed, redu), axis=0))
        diff3_neg = np.bitwise_and.reduce(~diff3_pos)
        diff2_cul = np.stack((diff3_pos, np.broadcast_to(diff3_neg, (diff3_pos.shape[0]))))

        # the diff has now been seperated into components, next is proceeding to generate the new branches
        assert self.size + passed.shape[0] <= self.limit, "out of states"
        ones = np.ones(passed.shape[0], dtype=self.emit_inner.dtype)
        aran = np.arange(self.used, self.used + passed.shape[0], dtype=self.emit_inner.dtype)
        new_gen = ones << aran # pyright: ignore[reportOperatorIssue]
        broad = np.broadcast_to(self.emit_inner[None, ...], (mask4.shape[0], self.count))
        reused = np.bitwise_and.reduce(broad, axis=1, where=mask4)
        res = np.concat((new_gen, reused), axis=0)
        self.size += diff2_cul.shape[0]

        print(diff3_neg)
        return diff2_cul, res

    def apply(self, match: np.ndarray, emit: Tokens) -> None:
        assert match.shape[1] == emit.shape[0], "mismatched arrays"
        assert match.shape[0] == 2, "match is malformed"

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
