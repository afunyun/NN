import numpy as np
from typing import Optional

gen = np.random.default_rng()
randarray = lambda size, dtype: gen.integers(np.iinfo(dtype).max, size=size, dtype=dtype)


class Neurray:
    def __init__(self, neurons:int, idtype=np.uint8, odtype=np.uint8):
        self.neurons = neurons
        self.idtype = idtype
        self.odtype = odtype

        self.training = False

        self.match = np.empty((2, neurons, 1), dtype=idtype)
        self.emit = np.empty((2, neurons, 1), dtype=odtype)
        self.choices = np.empty((2, neurons, 1), dtype=bool)

    def set_training(self):
        self.training = True

    def init_array(self, *args):
        # either pass in the arrays or generate some
        # as if anyone else besides myself knows what a better than random array is
        if not len(args):
            self.match[...] = randarray((2, self.neurons, 1), self.idtype)
            self.emit[...] = randarray((2, self.neurons, 1), self.odtype)
        else:
            self.match[...] = args[0]
            self.emit[...] = args[1]

    def forward(self, inputs:np.ndarray) -> np.ndarray:

        # check for resonance (saving for backwards)
        self.choices[0] = inputs & self.match[0] == 0
        self.choices[1] = inputs ^ self.match[1] >= self.match[1]

        # invert mask based on choice made
        hidden_states = np.where(self.choices, self.emit, ~self.emit)

        outputs = hidden_states[0] ^ hidden_states[1]
        
        if self.training:
            # Debug neuron state
            match_states = np.where(self.choices, ~self.match, self.match)
            match_loss_mask = ((~match_states ^ (inputs ^ match_states)) ^ ~inputs) < (match_states ^ inputs)
            if match_loss_mask[0] | match_loss_mask[1]:
                print(f"{inputs} is contained")
            else:
                print(f"{inputs} is not contained")
            if match_loss_mask[0] & match_loss_mask[1]:
                print(f"{inputs} is undiscovered for this neuron")
            else:
                print(f"{inputs} is trained for this neuron")

            # capture for backwards pass
            self.givens = inputs
            self.results = outputs

        # Outputs are not merged into a single value and should be treated as masks on previous steps
        return outputs

    def backward(self, target:np.ndarray, neurons:Optional[np.ndarray]=None) -> None:
        assert self.training, "NN Inputs and Outputs are non existant, place this class in training mode first!"
        
        batch_size = self.results.shape[1]
        # check if neurons have already converged, inverting the result to use as a mask
        output_diff = self.results ^ target
        target = np.copy(target)
        
        # TODO isolate search to neurons if present
        #neuron_convergance = (output_diff[...,0] != 0)

        # apply overide
        #if not neurons is None:
        #    neuron_convergance |= neruons

        # invert effects of matches that got inverted during forward pass
        match_states = np.where(self.choices, ~self.match, self.match)
        emit_states = np.where(self.choices, ~self.emit, self.emit)
        
        # Calculate diffs between expected and actual outputs
        expected_match = ~(self.givens ^ match_states)
        expected_emit  = (~(self.results ^ target)).reshape(1, *self.results.shape)

        emit_loss_mask = (~(emit_states ^ expected_emit) ^ target) < (emit_states ^ target)
        
        match_loss_mask = (~(match_states ^ expected_match) ^ self.givens) < (match_states ^ self.givens)

        # update the neurons
        self.match[...,0,None] = np.where(match_loss_mask, ~(self.match ^ expected_match), self.match)
        self.emit[...,0,None]  = np.where(emit_loss_mask,  ~(self.emit ^ expected_emit), self.emit)



class Elivrge:
    def __init__(self, step_count:int):
        self.steps = None 

    @property
    def base(self) -> np.ndarray:
        return np.zeros((self.neurons, 1), dtype=self.dtype)

    def forward(self, array):
        batch_size = array.shape[1]


