import numpy as np
from time import time

from neurray3 import Neurray

if __name__ == "__main__":
    args = (1, np.uint8, np.uint8)
    args2 = (1,)
    eliv = Neurray(*args)
    given = np.empty((args[0],args2[0]), dtype=args[1])
    target = np.empty((args[0],args2[0]), dtype=args[2])
    
    prev_result = None
    generations = 0
    mutations = 0

    given[...,0] = 5
    target[...,0] = 5

    eliv.init_array()
    eliv.set_training()


    start = time()
    #for i in range(500):
    while 1:
    #if 1:
        result = eliv.forward(given)
        generations += 1
        if not prev_result is None and np.equal(result, target).all():
            #pass
            break
        else:
            prev_result = result
        eliv.backward(target)

    end = time()
    print(f"""{args[0]} neuron{'s' if args[0]-1 else ''} | {args2[0]} target{'s' if args2[0]-1 else ''} | {generations} pass{'es' if generations-1 else ''} | {round((end - start) * 1_000, 2)} ms""")
    eliv.forward(given)
    print(eliv.forward(given-1))
   
    
