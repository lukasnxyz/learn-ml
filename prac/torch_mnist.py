import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 hidden layers
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
    Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)

    print(X_train.shape)
    print(Y_train.shape)

    import matplotlib.pyplot as plt
    #plt.imshow(X_train[1])
    #plt.show()

    model = MNIST()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1
    batch_size = 6

    for epoch in (t := trange(epochs)):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            print(X_batch.shape)
            Y_pred = model(X_batch)
            Y_batch = Y_train[i:i+batch_size]

            print(Y_batch[2])
            plt.imshow(X_batch[2].reshape(28,28))
            plt.show()

            print("ypred:", Y_pred.shape)
            print(Y_pred)
            Y_pred_n = []
            for y in Y_pred:
                y = y.clone().detach().numpy()
                print(y)
                Y_pred_n.append(y.argmax())

            Y_pred_n = torch.tensor(Y_pred_n, dtype=torch.float32)
            print(Y_pred_n)

            loss = loss_fn(Y_pred_n, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t.set_description("loss %.2f" % (loss))

    with torch.no_grad():
        y_pred = model(X)

    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy: {accuracy}")
