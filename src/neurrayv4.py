import numpy as np

from neurtypes import Case, State, Diff

class U1XToU1X:
    def __init__(self, match_slot:Case, cases:int) -> None:
        assert len(match_slot.shape) == 1, "malformed case"

        self.array_size = cases
        self.array_used = 0

        self.match = np.zeros((2, match_slot.shape[0], self.array_size), dtype=match_slot.dtype)

    # This is forward + reverse passes
    def forward(self, tokens:State) -> Diff:
        assert tokens.dtype == self.match.dtype
        assert tokens.shape[1] == self.match.shape[1]


        match_view = self.match[None, ..., :self.array_used]

        tokens_inv = ~tokens.copy()
        input: Diff = np.stack((tokens, tokens_inv), axis=1)

        # FORWARD
        selection = input[..., None] & match_view

        choices: np.ndarray = (selection == match_view).all((1,2))

        # REVERSE
        match_broad = np.broadcast_to(match_view, (choices.shape[0], 2, self.match.shape[1], self.array_used))
        reverse: Diff = np.bitwise_or.reduce(match_broad, axis=3, where=choices[:, None, None, :])

        diff: Diff = reverse & input ^ input
        # removing [[0, ..., 0][0, ..., 0]] instances, because harmful to compute
        diff_cul: Diff = diff[diff[:, 0].any(axis=1)]

        # dedupe as assign will blindly add multipule indentical cases otherwise
        diff_cul2: Diff = np.unique(diff_cul, axis=0)

        return diff_cul2

    # This is assign and apply
    def assign(self, diff:Diff) -> None:
        assert diff.shape[1] == 2, "diff is malformed"
        assert diff.shape[2] == self.match.shape[1], "case is mismatched"


        # ASSIGN
        diff2 = diff[..., None] ^ self.match[None, ..., :self.array_used]

        diffed_neg_mask = diff2[:, 0].any(axis=1).all(axis=0)
        new_case_mask = diff2[:, 0].any(axis=1).all(axis=1)

        neg_case: np.ndarray = np.bitwise_and.reduce(diff[:, 1], axis=0)

        # APPLY
        self.match[..., :self.array_used][1, :, ~diffed_neg_mask] &= neg_case

        new_cases = diff[new_case_mask]

        assert self.array_used + new_cases.shape[0] <= self.array_size
        self.match[..., self.array_used:self.array_used+new_cases.shape[0]] = np.permute_dims(new_cases, (1,2,0))
        self.match[1, :, self.array_used:self.array_used+new_cases.shape[0]] &= neg_case[..., None]

        self.array_used += new_cases.shape[0]
