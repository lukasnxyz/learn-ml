## Notes
- The difference between an ml engineer and a software engineer is that the software engineer writes
  the algorithms and such for the libraries that the ml engineers use. I want to do both. I don't
  want to be only an ml engineer (data scientist), because that's boring. I want to be an ml
  software engineer. That sounds much more exciting. And want to build and deploy homemade ml
  algorithms.
- Finding a good dataset to train on is very difficult, the best one so far is
  [this one](https://www.kaggle.com/code/martandsay/height-weight-regression-classification/input)
- ML dev stack: python, jupyter-notebook, numpy, pandas, matplotlib
- I need to learn how to debug matrix shape mismatches (it's very daunting right now)
- See https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%2013%20SVM.md#importing-the-dataset
  to properly import a dataset from a csv file
- Use more asserts!
- Feed forward, weighted linear combination, activation, regularisation, loss function, back propagation, gradient decent
- Always write an accompanying document report on a project once finished?

### What is?
- [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Relu](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
- [MNIST - The Hello World of ML](https://en.wikipedia.org/wiki/MNIST_database)
- [Adam optimizer](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

### General notes
- Progress bar in python: tqdm
- A tensor is just a representation of a scalar/vector/matrix/etc. so an object with dimension and shape
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

### Learning steps (bottom up)
Types of ML
- Supervised algorithms (regression, classification)
- Unsupervised and semi-supervised algorithms (clustering, dimensionality reduction, graph-based algorithms)
- Deep learning (CNNs and RNNs)
- Reinforcement learning (dynamic programming, Monte Carlo methods, heuristic methods)

Different tasks
- Computer vision (e.g.: classification, object detection, semantic segmentation)
- Natural language processing (e.g.: text classification, sentiment analysis, language modeling, machine translation)
- Recommending systems
- Classic machine learning

### Ideas I want to build
- An ML library
    - basic algo's
    - NN's
- Chatbot (language model)
