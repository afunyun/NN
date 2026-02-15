import numpy as np

from neurrayv4 import U1XToU1X

def sanity_test():

    eliv = U1XToU1X(np.empty(4, dtype=np.uint8), cases=6)

    temp = np.array([[8, 0, 0, 0], [2,0,0,0], [4,0,0,0]], dtype=np.uint8)

    rev = eliv.forward(temp)
    eliv.assign(rev)
    print(eliv.array_used) # 3 cases

    rev = eliv.forward(np.array([[9, 0, 0, 0], [12,0,0,0], [14,0,0,0]], dtype=np.uint8))
    eliv.assign(rev)
    print(eliv.array_used) # 4 cases (adding [1, 0, 0, 0])

    rev = eliv.forward(np.array([[9, 0, 0, 0], [12,0,0,0], [14,0,0,0]], dtype=np.uint8))
    print(eliv.array_used) # 4 cases (It's identical after all)



def dataset():
    import tensorflow_datasets as tfds

    # load full dataset
    mnist, info = tfds.load(
        "mnist",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        batch_size=-1,   # <- load everything as big tensors
    )

    # convert to numpy
    train_np, test_np = tfds.as_numpy(mnist)

    # now these are plain numpy arrays
    x_train, y_train = train_np
    x_test, y_test   = test_np

    def chunk_mnist_for_U1X(x: np.ndarray) -> np.ndarray:
        """
        x: numpy array of shape (N, 28, 28)
        returns: (N, 4, 4, 7, 7)
        """
        N, H, W, _ = x.shape
        assert H == 28 and W == 28

        # reshape + transpose trick
        tiles = (
            x
            .reshape(N, 4, 7, 4, 7)
            .transpose(0, 1, 3, 2, 4)
        )

        # flatten 7x7
        tiles = tiles.reshape(tiles.shape[0], 4, 4, -1)    # (N,4,4,49)

        # reorder to tokens
        tiles = tiles.transpose(0, 3, 1, 2)                # (N,49,4,4)

        # now flatten 4x4 to 16 tokens
        tiles = tiles.reshape(tiles.shape[0], tiles.shape[1], -1)  # (N,49,16)

        return tiles

    def recast_u64(a:np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(a)

        b = a.view(np.uint64)
        b = b.reshape(*a.shape[:-1], 2)
        return b

    # `tiles` is from your chunking function, shape (N, 4, 4, 7, 7)
    tiles_train = chunk_mnist_for_U1X(x_train)
    tiles_eval = chunk_mnist_for_U1X(x_test)

    tiles_train = recast_u64(tiles_train)
    tiles_eval = recast_u64(tiles_eval)

    all_tiles_train = tiles_train.reshape(tiles_train.shape[0] * tiles_train.shape[1], tiles_train.shape[2])
    all_tiles = np.unique(all_tiles_train, axis=0)
    print(f"{all_tiles.shape[0]} total unique tiles")
    # 1972878 tiles as is

    # we love setup being 4 seconds out of 28 second runtime on the poor laptop

    neur = U1XToU1X(np.empty(tiles_train.shape[2], tiles_train.dtype), cases=100_000) # case count inflated as chunking code was swapped for 7x7 tiles instead of 4x4

    counter = 0

    for tile in tiles_train:
        if counter % 1000 == 0:
            print(f"training at: {counter}")
        rev = neur.forward(tile)
        try:
            neur.assign(rev)
        except AssertionError:
            print(f"Broke at: {counter}, ran out out of cases\n")
            raise
        counter += 1
    print(f"\n\n{neur.array_used} cases used!\nMoving into validation\n")
    # 811 as is

    counter = 0
    misses = 0
    for tile in tiles_eval:
        if counter % 1000 == 0:
            print(f"eval at: {counter}")
        rev = neur.forward(tile)
        misses += rev.shape[0]
        counter += 1

    print(f"\n\n{misses} inputs were unable to be mapped for eval")
    # 0 as is




if __name__ == "__main__":
    dataset()
