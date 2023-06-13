from matplotlib.image import imread
from random import randrange
import numpy
import os

directory = os.fsencode("hotdogs")

def cost(w):
    result = 0.0
    for file in os.listdir(directory):
        filename = "hotdogs/" + os.fsdecode(file)
        image = imread(filename)

        x = image
        y = numpy.dot(w, x)
        d = y - 1
        result += d

    result /= len(os.listdir(directory))

    return result

def main():
    w = randrange(10)
    h = 1e-3
    rate = 1e-2

    for i in range(500):
        dconst = (cost(w + h) - cost(w)) / h
        w -= rate*dconst
        print("cost:", cost(w), "w:", w)

    print("----------------------")
    for i in range(len(tSet)):
        print("acual:", w*tSet[i][0], "expected:", tSet[i][1])

if __name__ == "__main__":
    main()
