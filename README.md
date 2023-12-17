## ml
For now this is just me learning about neural networks and machine learning, but I hope to soon turn this into a large
community project.

This is a library is meant for building and training neural networks in C. All implemented code has been written from
scratch and uses only [C standard libraries](https://en.cppreference.com/w/c/header) so as to keep this project as clean and minimal as possible. This neural
network library is meant to be portable and light weight to make machine learning more accessible to everyone.

### Quick start
For neural network (plan on turning it into a library)
```bash
$ make
$ ./bin/nn
```

### Todo
- Implement .clang-format
- Build minimal test system
- Build progress bar
- Build CSV parser

<details>
  <summary>Some other notes</summary>

  - https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab
  - https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
  - https://www.youtube.com/watch?v=w8yWXqWQYmU
  - https://avi.alkalay.net/2018/07/fedora-jupyter-notebook.html
  - https://en.wikipedia.org/wiki/MNIST_database
  - tqdm (progress bar)
  - https://www.kaggle.com/datasets/hojjatk/mnist-dataset
  - https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
  - Bitcoin Historical Data: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/

  - Gradient descent is basic machine learning algo
  - y = w(x) OR y = w(x) - b
  - goal -> w(x) - b = 0
  - square result from cost function to get more amplified result
  - input is usually a feature vector
  	1. Any value for w
  	2. Give to w to cost function to get prediction precision (close to 0, the more precise and accurate)
  	3. w - derivative of cost function (limit as h->0)
  	4. Apply learning rate
  	5. Iterate many times

</details>
