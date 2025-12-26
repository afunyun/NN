import numpy as np

from neurtypes import MatchBase, Neurray, Tokens, tokens_to_neurray

class U1XToU1X(MatchBase):
    def forward(self, input_state: Tokens) -> Tokens:
        assert input_state.dtype == self.match.dtype
        input: Neurray = tokens_to_neurray(input_state)
        choices = np.bitwise_or(input, self.match).all(axis=0)
        output: Tokens = np.bitwise_or.reduce(self.emit, axis=0, where=choices)
        return output
    # welcome back UXtoUX
    def reverse(self, output_state: Tokens) -> tuple[Tokens, Tokens]:
        assert output_state.dtype == self.emit.dtype
        choices = np.bitwise_or(output_state, self.emit) == output_state
        existant: Tokens = np.bitwise_or.reduce(self.match_inner[0], axis=0, where=choices)
        absent: Tokens = np.bitwise_or.reduce(self.match_inner[1], axis=0, where=choices)
        return existant, absent

    def backwards(self, input_state: Tokens, parsed: tuple[Tokens, Tokens], result: Tokens) -> None:
        pass
