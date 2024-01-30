#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# following guide:
# https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

# 3 layer nn with size 8 (features) input and size 1 output (class of 0 or 1)
class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

# a sequential neural network (sequence of layers)
if __name__ == "__main__":
    dataset = np.loadtxt("../data/pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:, 0:8]
    y = dataset[:, 8] # last column
    print("y:", y)

    # convert numpy float64 matricies to torch float32 tensors(matricies)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(X, dtype=torch.float32).reshape(-1, 1) # torch prefers n x 1 matricies instead of vectors

    model = PimaClassifier()
    loss_fn = nn.BCELoss() # binary cross entropy (loss function)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # adam is a better version of gradient descent

    epochs = 100
    batch_size = 10

    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            print("Xbatch:", Xbatch)
            print("y_pred:", y_pred)
            print("ybatch:", ybatch)

            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}, loss: {loss}")
