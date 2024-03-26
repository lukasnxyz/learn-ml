import numpy as np
from tqdm import trange
import csv
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def accuracy(pred, true):
	return np.sum(pred == true) / len(true)

class NET(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.h1 = nn.Linear(784, 128)
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

def main():
	with open("train.csv", "r") as file:
		next(file)
		reader = csv.reader(file)
		data_csv = list(reader)

	data = np.array(data_csv)
	data = data.astype(float)
	X = data[:, 1:]
	Y = data[:, 0]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)
	X_train = X_train.reshape(X_train.shape[0], -1)
	X_test = X_test.reshape(X_test.shape[0], -1)

	X_train = torch.tensor(X_train, dtype=torch.float32)
	X_test = torch.tensor(X_test, dtype=torch.float32)
	Y_train = torch.tensor(Y_train, dtype=torch.long)

	model = NET()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	epochs = 1
	batch_size = 32

	train_dataset = TensorDataset(X_train, Y_train)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	for epoch in (t := trange(epochs)):
		for X_batch, Y_batch in train_loader:
			# forward pass
			out = model(X_batch)
			optimizer.zero_grad()
			loss = loss_fn(out, Y_batch)

			# backward pass
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

	# test.csv
	with open("test.csv", "r") as file:
		next(file)
		reader = csv.reader(file)
		data_csv = list(reader)

	data = np.array(data_csv)
	data = data.astype(float)
	X = data[:, :]
	X = torch.tensor(X, dtype=torch.float32)
	predictions = model(X)

	with open(r"submission.csv", "a") as file:
		writer=csv.writer(file, lineterminator='\n')
		for i, pred in enumerate(predictions, start=1):
			pred = pred.clone().detach().numpy()
			val = pred.argmax()
			row = [i, val]
			writer.writerow(row)

if __name__ == "__main__":
	main()
