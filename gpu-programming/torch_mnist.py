import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

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

  def forward(self, X):
    X = self.act1(self.h1(X))
    X = self.act2(self.h2(X))
    X = self.output(X)

    return X

if __name__ == "__main__":
  df = pd.read_csv("data/mnist_data.csv")

  # first column = labels, rest = pixels
  Y = df.iloc[:, 0].values
  X = df.iloc[:, 1:].values.astype(np.float32)

  # normalize pixel values to [0,1]
  X /= 255.0

  indices = np.arange(len(X))
  np.random.shuffle(indices)

  split = int(0.75 * len(X))
  train_idx, test_idx = indices[:split], indices[split:]

  X_train, X_test = X[train_idx], X[test_idx]
  Y_train, Y_test = Y[train_idx], Y[test_idx]

  X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
  X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
  Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)
  Y_test = torch.tensor(Y_test, dtype=torch.long).to(device)

  model = MNIST().to(device)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)

  epochs = 5
  batch_size = 32

  for epoch in (t := trange(epochs)):
    for i in range(0, len(X_train), batch_size):
      # batching
      X_batch = X_train[i:i+batch_size]
      Y_batch = Y_train[i:i+batch_size]

      # forward pass
      out = model(X_batch)
      optimizer.zero_grad()
      loss = loss_fn(out, Y_batch)

      # backward pass
      loss.backward()
      optimizer.step()

      t.set_description("loss %.2f" % (loss.item()))

  # make predictions
  predictions = model(X_test)
  preds = []
  for pred in predictions:
    pred = pred.clone().detach().cpu().numpy()
    val = pred.argmax()
    preds.append(val)

  a = accuracy(preds, Y_test.cpu().numpy()) * 100
  print(f"Accuracy: %.2f%%" % (a))
