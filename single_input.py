from random import random, Random, randrange
from sys import maxsize

def main():
    data = [[0, 0],
            [1, 2],
            [2, 4],
            [3, 6],
            [4, 8]]

    seed = randrange(maxsize)
    rng = Random(seed)

    w = random() * 10.0
    h = 1e-3
    rate = 1e-2

    print(w)

if __name__ == "__main__":
    main()
