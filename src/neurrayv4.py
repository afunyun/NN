import numpy as np

from neurtypes import Case, State, Diff

class U1XToU1X:
    def __init__(self, match_slot:Case, cases:int) -> None:
        assert len(match_slot.shape) == 1, "malformed case"

        self.array_size = cases
        self.array_used = 0

        self.match = np.zeros((2, match_slot.shape[0], self.array_size), dtype=match_slot.dtype)

        # Debug stats
        self.debug_case_activations = np.zeros(self.array_size, dtype=np.uint64)
        self.reset_debug_stats()

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

        if self.array_used:
            case_hits = np.count_nonzero(choices, axis=0).astype(np.uint64, copy=False)
            self.debug_case_activations[:self.array_used] += case_hits
            self.debug_total_case_activations += int(case_hits.sum())

        # REVERSE
        match_broad = np.broadcast_to(match_view, (choices.shape[0], 2, self.match.shape[1], self.array_used))
        reverse: Diff = np.bitwise_or.reduce(match_broad, axis=3, where=choices[:, None, None, :])

        diff: Diff = reverse & input ^ input
        # removing [[0, ..., 0][0, ..., 0]] instances, because harmful to compute
        diff_cul: Diff = diff[diff[:, 0].any(axis=1)]

        # dedupe as assign will blindly add multipule indentical cases otherwise
        diff_cul2: Diff = np.unique(diff_cul, axis=0)

        self.debug_forward_calls += 1
        self.debug_total_inputs += int(tokens.shape[0])
        self.debug_total_diffs += int(diff_cul2.shape[0])

        return diff_cul2

    def debug_snapshot(self) -> dict[str, float | int]:
        avg_diffs_per_forward = 0.0
        avg_diffs_per_input = 0.0
        avg_case_activations_per_input = 0.0

        if self.debug_forward_calls:
            avg_diffs_per_forward = self.debug_total_diffs / self.debug_forward_calls
        if self.debug_total_inputs:
            avg_diffs_per_input = self.debug_total_diffs / self.debug_total_inputs
            avg_case_activations_per_input = self.debug_total_case_activations / self.debug_total_inputs

        active_case_activations = self.debug_case_activations[:self.array_used]
        case_activation_mean = 0.0
        if self.array_used:
            case_activation_mean = float(active_case_activations.mean())

        return {
            "forward_calls": self.debug_forward_calls,
            "total_inputs": self.debug_total_inputs,
            "total_diffs": self.debug_total_diffs,
            "total_case_activations": self.debug_total_case_activations,
            "active_cases": self.array_used,
            "avg_diffs_per_forward": avg_diffs_per_forward,
            "avg_diffs_per_input": avg_diffs_per_input,
            "avg_case_activations_per_input": avg_case_activations_per_input,
            "mean_activations_per_case": case_activation_mean,
        }

    def debug_case_activation_counts(self) -> np.ndarray:
        return self.debug_case_activations[:self.array_used].copy()

    def reset_debug_stats(self) -> None:
        self.debug_forward_calls = 0
        self.debug_total_inputs = 0
        self.debug_total_diffs = 0
        self.debug_total_case_activations = 0
        self.debug_case_activations.fill(0)

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
