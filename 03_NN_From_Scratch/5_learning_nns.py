import numpy as np

np.random.seed(0)

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

def unit_step(x):
    return 0 if x < 0 else 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, X):
        self.output = np.dot(X, self.weights) + self.biases

class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

if __name__ == "__main__":
    X, y = spiral_data(100, 3)

    layer1 = Layer_Dense(2, 5) # number of inputs, number of neurons
    activation1 = Activation_Relu()

    layer1.forward(X)
    activation1.forward(layer1.output)
    print(activation1.output)

    '''
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
    plt.show()
    '''
