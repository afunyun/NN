import numpy as np

from neurrayv4 import U1XToU1X

def sanity_test():

    eliv = U1XToU1X(np.empty(4, dtype=np.uint8), np.uint8, cases=6)

    temp = np.array([[8, 0, 0, 0], [2,0,0,0], [4,0,0,0]], dtype=np.uint8)
    print(temp.shape)

    res, rev = eliv.forward(temp)
    eliv.assign(rev)

    res, rev = eliv.forward(np.array([[9, 0, 0, 0], [12,0,0,0], [14,0,0,0]], dtype=np.uint8))
    eliv.assign(rev)

    res, rev = eliv.forward(np.array([[9, 0, 0, 0], [12,0,0,0], [14,0,0,0]], dtype=np.uint8))
    print(res)

    print(eliv.match, eliv.array_used)


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

    def chunk_mnist_for_U1X(x):
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

    # `tiles` is from your chunking function, shape (N, 4, 4, 7, 7)
    tiles_train = chunk_mnist_for_U1X(x_train)
    tiles_eval = chunk_mnist_for_U1X(x_test)

    neur = U1XToU1X(np.empty(tiles_train.shape[2], np.uint8), np.uint64)

    counter = 0
    for tile in tiles_train:
        if counter % 1000 == 0:
            print(f"training at: {counter}")
        _, rev = neur.forward(tile)
        try:
            neur.assign(rev)
        except AssertionError:
            print(f"Broke at {counter}")
            break
        else:
            counter += 1
    print(f"\n\n{neur.array_used} cases used!\nMoving into validation\n")

    counter = 0
    misses = 0
    for tile in tiles_eval:
        if counter % 1000 == 0:
            print(f"training at: {counter}")
        _, rev = neur.forward(tile)
        misses += rev.shape[0]
        counter += 1

    print(f"\n\n{misses} inputs were unable to be mapped for eval")




if __name__ == "__main__":
    dataset()
