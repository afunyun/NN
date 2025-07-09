import numpy as np

class MaskToMask:
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
        for n in range(self.neurons-2):
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
        mask = (self.output == 0)
        none = self.output[mask]
        none_ex = target[mask]
        none_gi = self.givens[0, mask]
        some = self.output[~mask]
        some_ex = target[~mask]
        some_gi = self.givens[0, ~mask]
        
        matched = self.choices[..., mask]

        if (cases := none.shape[0]) > 0:
            # we need to add a new case here
            if cases + self.count > self.neurons:
                # TODO expand instead
                raise RuntimeError("Out of slots")
            for slot in range(cases):
                self.match[self.count + slot] = none_gi[slot]
                self.emit[self.count + slot] = none_ex[slot]
            self.count += cases
        if (cases := some.shape[0]) > 0:
            for slot in range(cases):
                self.match = np.choose(matched[...,slot].reshape(self.neurons,1), (self.match & some_gi[slot], self.match))
                self.emit = self.emit & some_ex[slot]

nums = [
0b11100000,
0b00011100,
0b10000011,
0b11001100,
0b10100101,
0b01011010,
0b11110000,
0b00001111,
0b11000011,
0b11111000]

# o7 Elivrge
eliv = MaskToMask(16, np.uint8, np.uint8)
eliv.init_array()
eliv.set_training()
array = np.array(nums, np.uint8)
res = eliv.forward(array[0:5].reshape(1,5))
eliv.backward(array[0:5])
#eliv.forward(array[6:9].reshape(1,3))
#eliv.backward(array[6:9])
res = eliv.forward(array[0:5].reshape(1,5))
eliv.backward(array[0:5])
print(res)
print(eliv.match, eliv.emit)





