## ml
For now this is just me learning about neural networks and machine learning, but I hope to soon turn this into a large
community project.

This library is meant for building and training neural networks in C. All implemented code has been written from
scratch and uses only [C standard libraries](https://en.cppreference.com/w/c/header) so as to keep this project as clean and minimal as possible. This neural
network library is meant to be portable and light weight to make machine learning more accessible to everyone on any
computer. I also want to do this in C because it's far more fun than in Python.

### Quick start
For neural network
```bash
$ make
$ ./bin/nn
```
To run tests
```bash
$ make test
```

### Todo
- [ ] Implement .clang-format
- [ ] Build progress bar (header)
- [ ] Build CSV parser (header)
- [ ] la to tensor, keep la (header)
- [ ] Neural network using tensors (header)

<details>
  <summary>Some other notes</summary>

  #### Learning
  - Neural networks tutorial playlist: https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
  - Convolutional neural network: https://en.wikipedia.org/wiki/Convolutional_neural_network
  - Relu: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
  - Mnist classifier from scratch: https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab
  - nn from scatch in python/numpy: https://www.youtube.com/watch?v=w8yWXqWQYmU
  - What's mnist: https://en.wikipedia.org/wiki/MNIST_database

  #### Data
  - Mnist dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
  - Bitcoin Historical Data: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data/

  #### Notes
  - Progress bar in python: tqdm
  - A tensor is just a representaiton of a scalar/vector/matrix/etc. so an object with dimension and shape
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
