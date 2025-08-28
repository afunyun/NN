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


class Bwaa:
    # TODO make these work with more than 64 bits
    def __init__(self, isize, osize, neurons):
        idtype = size_to_dtype(isize)
        odtype = size_to_dtype(osize)
        self.neurons = neurons
        self.count = 0
        
        self.training = False

        self.match = np.zeros((neurons, 1), dtype=idtype)
        self.emit = np.zeros((neurons, 1), dtype=odtype)

    def set_training(self):
        self.training = True

    def override_array(self, args):
        if len(args) == 2:
            self.match[...] = args[0]
            self.emit[...] = args[1]

class U1XToU1X(Bwaa):
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
        # TODO bug where changes don't propigate when a CONTAINS is split into two smaller parts
        # might also cause duplication?
        
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

class UXToUX(Bwaa):
    def forward(self, inputs:np.array):
        # Surely there is a better way to collapse this down into a 1d array
        choices = self.match == inputs
        outputs = np.choose(choices, (0, self.emit))
        output = outputs[0]
        for n in range(self.neurons-1):
            output |= outputs[n+1]
        
        if self.training:
            # capture for backwards pass
            self.givens = inputs
            self.output = output

        return output

    def backward(self, target:np.array):
        assert self.training, "NN Inputs and Outputs are non existant, place this class in training mode first!"
        mask = self.output == 0
        none_ex = target[mask]
        none_gi = self.givens[mask]
        
        # This is either: EXACT or NEW

        # The #1 reason this doesn't work on GPU currently
        # All EXACT could work, anything else has no shot
        for slot, case in enumerate(none_ex):
            hard_mask = (case == self.emit)
            # EXACT
            if hard_mask.any():
                self.match[hard_mask] &= none_gi[slot]
            # NEW
            else:
                assert self.count < self.neurons, "Out of Slots"
                self.match[self.count] = none_gi[slot]
                self.emit[self.count] = case
                self.count += 1


class U1XToInc(Bwaa):
    def forward(self, inputs:np.array):
        choices = (self.match | inputs) == inputs
        outputs = np.choice(choices, (0, self.emit))

        output = np.sum(outputs, axis=1)

        if self.training:
            # capture for backwards pass
            self.givens = inputs
            self.output = output
        
        return output

    def backward(self, target:np.array):
        assert self.training, "NN Inputs and Outputs are non existant, place this class in training mode first!"
        mask = self.output == 0
        none_ex = target[mask]
        none_gi = self.givens[mask]
        
        for slot, case in enumerate(none_ex):
            hard_mask = 

        pass

# Honestly probably useless, U1XToInc can emulate this while being more flexable
class UXToInc(Bwaa):
    def forward(self, inputs:np.array):
        choices = (inputs >= self.match)
        outputs = np.choice(choices, (0, self.emit))

        output = np.sum(outputs, axis=1)

        if self.training:
            # capture for backwards pass
            self.givens = inputs
            self.output = output

    def backward(self, target:np.array):
        pass


"""
if a slot is 1 then it can't be reduced

What to do if I need 5->3 but all I have is 1,1,1,2
I can reduce the 2 -> 1, but that leaves me at 4

The real question is "is it a valid case?"
I'm on maybe. I think in that case it would have to bounce back.
That's a new API I'd have to support, increase the others by the amount I am remaining.
Probably don't touch this class in that case, odds are this is correct but the other one has scope creep.

Now for the more standard issue

We need to increase from 3->5 and we have 1,1,1
How do we distibute these?
I think bottom to top. so 1,2,2. The top case is older and probably already settled in.

That does not work when we have 3->7 instead with the same 1,1,1
That case would be equal to 2,2,3
But it could just as likely be that it needs to be 1,1,5 or 1,2,4.

I think I should waterfall with half+1
over adding it all onto the last one
I'll make an exception for +X and we have X cases that are each 1 off, in that case balence

Both of them seem like valid answers tho

Now for reduction in valid cases
I think this one has to be delete until 1 and keep carrying over.

The last thing to figure out is how this interacts with splitting
You can't really split these. You need 3 (A, B, A+B). Trying to balance these three is tricky.
On the other hand, I guess that'd simply down the allocation. Half to A and B and the other half to A+B

This does place a large barrier on of ever editing old cases, so it isn't perfect. On the other hand, as more exact the quarys are the more likely either of the apporaches will reach it.

Regardless of Bit Mask or Uint version, it still has to be this way.
"""






        



