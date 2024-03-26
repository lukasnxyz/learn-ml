import numpy as np
#import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

def accuracy(pred, true):
	return np.sum(pred == true) / len(true)

class CLASS(nn.Module):
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
	import pandas as pd
	from sklearn.model_selection import train_test_split

	td = pd.read_csv("train.csv")
	td = np.array(td)
	y = td[ :,1]
	X = td[ :2,]

	print(len(y))
	print(X)
	return

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)




	#X_train = X_train.reshape(X_train.shape[0], -1)
	#X_test = X_test.reshape(X_test.shape[0], -1)

	X_train = torch.tensor(X_train, dtype=torch.float32)
	X_test = torch.tensor(X_test, dtype=torch.float32)
	Y_train = torch.tensor(Y_train, dtype=torch.long)

	model = CLASS()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	epochs = 100
	batch_size = 100

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

if __name__ == "__main__":
	main()
