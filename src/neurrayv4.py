import numpy as np

from neurtypes import Rdu, case_gen_64, Case, State, Diff

class U1XToU1X:
    def __init__(self, match_slot:Case, emit_dtype:type[np.unsignedinteger], cases:int|None=None) -> None:
        bit_size: int = np.iinfo(emit_dtype).bits
        self.array_size: int = ((bit_size * (bit_size-1)) // 2) if cases is None else cases
        self.array_used = 0

        self.match = np.zeros((2, match_slot.shape[0], self.array_size), dtype=match_slot.dtype)
        self.emit = case_gen_64(emit_dtype)[..., :self.array_size]

    # This is forward + reverse passes
    def forward(self, tokens:State) -> tuple[Case, Diff]:
        assert tokens.dtype == self.match.dtype
        assert tokens.shape[1] == self.match.shape[1]

        match_view = self.match[None, ..., :self.array_used]
        emit_view = self.emit[None, ..., :self.array_used]

        tokens_inv = ~tokens.copy()
        input: Diff = np.stack((tokens, tokens_inv), axis=1)

        # FORWARD
        selection = input[..., None] & match_view
        choices = (selection == match_view).all(axis=(1, 2))

        emit_broad = np.broadcast_to(emit_view, (choices.shape[0], self.array_used))
        output: Case = emit_broad |Rdu(1)== choices

        # REVERSE
        match_broad = np.broadcast_to(match_view, (choices.shape[0], 2, self.match.shape[1], self.array_used))
        reverse: Diff = match_broad |Rdu(3)== choices[:, None, None, :]

        diff: Diff = reverse & input ^ input
        # removing [[0, ..., 0][0, ..., 0]] instances, because harmful to compute
        diff_cul: Diff = diff[diff[:, 0].any(axis=1)]

        return output, diff_cul

    # This is assign and apply
    def assign(self, diff:Diff) -> None:
        assert diff.shape[1] == 2, "diff is malformed"
        assert diff.shape[2] == self.match.shape[1], "case is mismatched"

        diff2 = diff[..., None] ^ self.match[None, ..., :self.array_used]

        diffed_neg_mask = diff2[:, 0].any(axis=1).all(axis=0)
        new_case_mask = diff2[:, 0].any(axis=1).all(axis=1)

        neg_case: np.ndarray = diff[:, 1] &Rdu(0)== True

        self.match[..., :self.array_used][1, :, ~diffed_neg_mask] &= neg_case

        new_cases = diff[new_case_mask]

        assert self.array_used + new_cases.shape[0] <= self.array_size
        self.match[..., self.array_used:self.array_used+new_cases.shape[0]] = np.permute_dims(new_cases, (1,2,0))
        self.match[1, :, self.array_used:new_cases.shape[0]+self.array_used] &= neg_case[..., None]

        self.array_used += new_cases.shape[0]
