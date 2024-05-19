import numpy as np

# Todo:
# In progress:
# Code review:

class Tensor:
    def __init__(self):
        pass

class NN:
    def __init__(self):
        pass

class Loss:
    pass

class Activation:
    pass

# for random practic dataset
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

if __name__ == "__main__":
    X, y = spiral_data(points=100, classes=3)

    # for _ in range(epochs):
        # batch
        # layer 1, activation 1
        # layer 2, activation 2
        # activation: y_pred

        # loss
        # gradient descent
        # loss backward pass
        # optomizer step

    # predictions and accuracy
