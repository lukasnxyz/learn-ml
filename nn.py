import numpy as np
import csv

# Basic neural network, consists of multidimensional arrays
# For gender will 1x2 vectors
class Tensor:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.rand(input_size, hidden_size).astype(float)
        self.bias_hidden = np.random.rand(hidden_size).astype(float)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size).astype(float)
        self.bias_output = np.random.rand(output_size).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # propagation
    def forward(self, inputs):
        self.inputs = inputs # Height and weight
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.activated_output = self.sigmoid(self.output)

        return self.activated_output

    # propagation
    def backward(self, target, learning_rate):
        # Compute gradients and update weights and biases using backpropagation
        output_error = target - self.activated_output
        delta_output = output_error * self.sigmoid_derivative(self.activated_output)

        hidden_error = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = hidden_error * self.sigmoid_derivative(self.hidden_output)

        self.hidden_output = np.array(self.hidden_output)
        np.transpose(self.hidden_output)

        delta_output = float(delta_output)

        #print("weight_input_hidden size: " + str(self.weights_input_hidden.shape))

        #print("delta_hidden shape: " + str(delta_hidden.shape))
        #print("inputs shape: " + str(self.inputs.shape))

        self.weights_hidden_output += (self.hidden_output.dot(delta_output) * learning_rate).reshape((2,1))
        self.bias_output += np.sum(delta_output) * learning_rate

        self.weights_input_hidden += self.inputs.T.dot(delta_hidden) * learning_rate
        self.bias_hidden += np.sum(delta_hidden) * learning_rate

    def train(self, input_data, target_data, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(input_data)):
                inputs = input_data[i]
                target = target_data[i]

                # Forward pass
                prediction = self.forward(inputs)

                # Backpropagation and weight updates
                self.backward(target, learning_rate)

                # Print loss for monitoring
                loss = np.mean(np.square(target - prediction))
                print(f"Epoch {epoch + 1}, Sample {i + 1}, Loss: {loss:.6f}")

# Move main to main.py once library is working
def main():
    '''
    data = []
    with open('data/500_Person_Gender_Height_Weight_Index.csv', 'r') as f: # 1 Male, 0 Female
        reader = csv.reader(f)
        for row in reader:
            data.append(row)

    data = np.array(data)
    '''

    # XOR is only true if the input differ
    data = [
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]
    ]

    input_size = 2  # Height and weight
    hidden_size = 2  # Number of hidden units
    output_size = 1  # Gender (0 or 1)(female, male)

    input_data = np.array([(x[0], x[1]) for x in data]).astype(float)
    target_data = np.array([np.array([x[2]]) for x in data]).astype(float)

    pred = Tensor(input_size, hidden_size, output_size)

    epochs = 1000
    rate = 1e-3

    pred.train(input_data, target_data, epochs, rate)

    #test_data = np.array([[185, 96], [149, 61]]) # 1(male), 0(female)
    test_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    predictions = pred.forward(test_data)

    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
