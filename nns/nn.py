import numpy as np

np.random.seed(0)

# implement back propagation using gradient descent algorithm

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

def relu(x):
    return np.max(0, x)

def sigmoid(x):
    return 1 / (1 - np.exp(-x))

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, X):
        self.output = np.dot(X, self.weights) + self.biases

class NN:
    def __init__(self, af=relu):
        pass

    def forward(self, X):
        pass

    def backward(self):
        pass

class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax: # softmax is used in last layer to predict class
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1: # mean scalar value were passed
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

if __name__ == "__main__":
    X, y = spiral_data(points=100, classes=3)

    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_Relu()

    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(activation2.output[:5])

    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output, y)

    print(loss)
