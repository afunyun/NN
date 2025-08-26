import numpy as np



class U1XToU1X:
    def __init__(self, neurons:int, idtype=np.uint8, odtype=np.uint8):
        self.neurons = neurons
        self.idtype = idtype
        self.odtype = odtype

        self.training = False

        self.match = np.empty((neurons, 1), dtype=idtype)
        self.emit = np.empty((neurons, 1), dtype=odtype)

    def set_training(self):
        self.training = True

    def init_array(self, *args):
        # either pass in the arrays or generate some
        # as if anyone else besides myself knows what a better than random array is
        if not len(args):
            self.match[...] = np.zeros((self.neurons, 1), self.idtype)
            self.emit[...] = np.zeros((self.neurons, 1), self.odtype)
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
        
        if (cases := none_ex.shape[0]) != 0:
            # This is either: EXACT, CONTAINS, or NEW
            for slot, case in zip(range(cases), none_ex):
                hard_mask = (case == self.emit)
                loose_mask = (case & self.emit) != 0
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
                else:
                    assert self.count < self.neurons, "Out of Slots"
                    self.match[self.count] = none_gi[slot]
                    self.emit[self.count] = case
                    self.count += 1




# case
# M 1 0
# 1 1 0
# 0 0 0






masks = np.array([
0b00000001,
0b00001001]
, dtype=np.uint8)

vals = np.array([
0b00000001,
0b00001001]
, dtype=np.uint8)

test = np.array([8], dtype=np.uint8)


# o7 Elivrge
eliv = U1XToU1X(6, np.uint8, np.uint8)
eliv.init_array()
eliv.set_training()
res = eliv.forward(masks)
print(res, vals)
eliv.backward(vals)
res = eliv.forward(masks)
print(res, vals)

res = eliv.forward(test)
print(res, 8)


print(eliv.match, eliv.emit)

