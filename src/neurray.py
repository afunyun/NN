import numpy as np

# TODO handle numbers inbetween these
def size_to_dtype(size:int):
    if size == 8:
        return np.uint8
    if size == 16:
        return np.uint16
    if size == 32:
        return np.uint32
    if size == 64:
        return np.uint64

class U1XToU1X:
    # TODO improve for more than 64 bits
    def __init__(self, isize=8, osize=8):
        idtype = size_to_dtype(isize)
        odtype = size_to_dtype(osize)
        self.neurons = osize

        self.training = False

        self.match = np.zeros((osize, 1), dtype=idtype)
        self.emit = np.zeros((osize, 1), dtype=odtype)

    def set_training(self):
        self.training = True

    def init_array(self, *args):
        # either pass in the arrays or generate some
        # as if anyone else besides myself knows what a better than random array is
        if not len(args):
            self.count = 0
        else:
            raise "bwaaaaa" # is an error itself
            self.match[...] = args[0]
            self.emit[...] = args[1]

    def forward(self, inputs:np.ndarray) -> np.ndarray:
        # check for resonance
        choices = (inputs | self.match) == inputs
        # Either it exists (gain what it isn't) or it doesn't (gain nothing)
        outputs = np.choose(choices, (0, self.emit))
        output = outputs[0]
        for n in range(self.neurons-1):
            output |= outputs[n+1]
        
        if self.training:
            # capture for backwards pass
            self.givens = inputs
            self.output = output
        
        return output

    def backward(self, target:np.ndarray) -> None:
        assert self.training, "NN Inputs and Outputs are non existant, place this class in training mode first!"
        mask = self.output == 0
        none_ex = target[mask]
        none_gi = self.givens[mask]
        
        # This is either: EXACT, CONTAINS, or NEW

        # The #1 reason this doesn't work on GPU currently
        # All EXACT could work, anything else has no shot
        for slot, case in enumerate(none_ex):
            hard_mask = (case == self.emit)
            loose_mask = (case & self.emit) != 0
            # EXACT
            if hard_mask.any():
                self.match[hard_mask] &= none_gi[slot]
            # CONTAINS
            elif loose_mask.any():
                new_match = self.match[loose_mask] ^ none_gi[slot]
                self.match[loose_mask] &= none_gi[slot]
                new_emit = self.emit[loose_mask] ^ none_ex[slot]
                self.emit[loose_mask] &= none_ex[slot]
                assert self.match.shape[0] > 1, "Identical Slots"
                assert self.count < self.neurons, "Out of Slots"
                self.match[self.count] = new_match
                self.emit[self.count] = new_emit
                self.count += 1
            # NEW
            else:
                assert self.count < self.neurons, "Out of Slots"
                self.match[self.count] = none_gi[slot]
                self.emit[self.count] = case
                self.count += 1



