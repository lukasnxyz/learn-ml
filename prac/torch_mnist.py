import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

def accuracy(pred, true):
    return np.sum(pred == true) / len(true)

class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(784, 128) # 784 features = 28 * 28 image
        self.act1 = nn.ReLU()
        self.h2 = nn.Linear(128, 128)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(128, 10)
        self.act_output = nn.Sigmoid()

    def forward(self, X):
        X = self.act1(self.h1(X))
        X = self.act2(self.h2(X))
        X = self.act_output(self.output(X))

        return X

if __name__ == "__main__":
    from keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)

    model = MNIST()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 100
    batch_size = 32

    for epoch in (t := trange(epochs)):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            out = model(X_batch)
            optimizer.zero_grad()
            loss = loss_fn(out, Y_batch)
            loss.backward()
            optimizer.step()

        t.set_description("loss %.2f" % (loss))

    # make predictions
    predictions = model(X_test)
    preds = []
    for pred in predictions:
        pred = pred.clone().detach().numpy()
        val = pred.argmax()
        preds.append(val)

    a = accuracy(preds, Y_test) * 100
    print(f"Accuracy: %.2f%%" % (a))

    # graphing to show
    '''
    import matplotlib.pyplot as plt
    for _ in range(7):
        i = np.random.randint(len(preds))
        print(f"true: {Y_test[i]} predicted: {preds[i]}")
        plt.imshow(X_test[i].reshape(28, 28))
        plt.title(f"Prediction: {preds[i]}")
        plt.show()
    '''
