#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

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

    # convert numpy float64 matricies to torch float32 tensors(matricies)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1) # torch prefers n x 1 matricies instead of vectors

    model = PimaClassifier()
    loss_fn = nn.BCELoss() # binary cross entropy (loss function)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # adam is a better version of gradient descent

    epochs = 1000
    batch_size = 15

    for epoch in (t := trange(epochs)):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]

            print("pred:", y_pred.shape)
            print("batch:", ybatch.shape)
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t.set_description("loss %.2f" % (loss))


    X_test = [[6.0, 148.0, 72.0, 35.0, 0.0, 33.599998474121094, 0.6269999742507935, 50.0], # 1
            [1.0, 85.0, 66.0, 29.0, 0.0, 26.600000381469727, 0.35100001096725464, 31.0], # 0
            [8.0, 183.0, 64.0, 0.0, 0.0, 23.299999237060547, 0.671999990940094, 32.0], # 1
            [1.0, 89.0, 66.0, 23.0, 94.0, 28.100000381469727, 0.16699999570846558, 21.0], # 0
            [0.0, 137.0, 40.0, 35.0, 168.0, 43.099998474121094, 2.2880001068115234, 33.0]] # 1

    y_test = [[1], [0], [1], [0], [1]]

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # compute accuracy
    with torch.no_grad():
        y_pred = model(X)

    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy: {accuracy}")

    predictions = model(X_test)
    rounded = predictions.round()

    def accuracy(true, pred):
        accuracy = np.sum(true == pred) / len(true)
        return accuracy
    y_test, rounded = y_test.clone().detach(), rounded.clone().detach()
    a = accuracy(np.array(y_test), np.array(rounded))
    print("Accuracy: %.2f" % (a))
