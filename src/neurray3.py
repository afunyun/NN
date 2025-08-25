import numpy as np



class U1XToU1X:
    def __init__(self, neurons:int, idtype=np.uint8, odtype=np.uint8):
        self.neurons = neurons
        self.idtype = idtype
        self.odtype = odtype

        self.training = False

        self.match = np.empty((neurons, 1), dtype=idtype)
        self.emit = np.empty((neurons, 1), dtype=odtype)
        self.choices = np.empty((1, neurons, 1), dtype=bool)
        # Mask needs to be all ones, so that &ing it passes through prev step
        self.mask = np.array((np.iinfo(odtype).max), dtype=odtype).reshape((1,1))

    def set_training(self):
        # Extra space for exact clause for training
        # TODO self.choices should just be a bit array (exact + contained) to not waste storage
        self.choices = np.empty((2, self.neurons, 1), dtype=bool)
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
        # check for resonance (saving for backwards)
        self.choices = (inputs | self.match) == inputs
        # Either it exists (gain what it isn't) or it doesn't (gain nothing)
        outputs = np.choose(self.choices, (0, self.emit))
        output = outputs[0]
        for n in range(self.neurons-1):
            output |= outputs[n+1]
        
        if self.training:
            # exact match?
            self.choices[1] = inputs == self.match[0]
            # capture for backwards pass
            self.givens = inputs
            self.results = outputs
            self.output = output
        
        return output

    def backward(self, target:np.ndarray) -> None:
        assert self.training, "NN Inputs and Outputs are non existant, place this class in training mode first!"
        mask = self.output == 0
        none = self.output[mask]
        none_ex = target[mask]
        none_gi = self.givens[0, mask]
        
        matched = self.choices[..., ~mask]
        if (cases := none.shape[0]) > 0:
            for slot, case in zip(range(cases), none_ex):
                mask = (case & self.emit) != 0
                print(mask)
                if mask.any():
                    print(slot)
                    self.match[mask] &= none_gi[slot]
                else:
                    if self.count < self.neurons:
                        self.match[self.count] = none_gi[slot]
                        self.emit[self.count] = case
                        self.count += 1
                    else:
                        raise RuntimeError("Out of slots")




# case
# M 1 0
# 1 1 0
# 0 0 0






masks = np.array([
0b01001001,
0b00101001,
0b00000110,
0b01100110,
0b11000000,
0b10000000,
0b00010000,
0b00010010,
0b00010001]
, dtype=np.uint8)

vals = np.array([
0b00000001,
0b00000001,
0b00000010,
0b00000010,
0b00000100,
0b00000100,
0b00001000,
0b00001010,
0b00001001]
, dtype=np.uint8)


# o7 Elivrge
eliv = U1XToU1X(6, np.uint8, np.uint8)
eliv.init_array()
eliv.set_training()
res = eliv.forward(masks.reshape(1,9))
print(res, vals)
eliv.backward(vals)
res = eliv.forward(masks.reshape(1,9))
print(res, vals)

print(eliv.match, eliv.emit)

