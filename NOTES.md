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

### Tutorials
- [NNs from zero to hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [NN from scatch in python/numpy](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- [MNIST classifier from scratch article](https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab)
- [MNIST classifier from scratch](https://github.com/kdexd/digit-classifier)
- [Stanford CS229 ML course](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
- [Python refresher](https://learnxinyminutes.com/docs/python/)
- [Full course Intro to ML](https://www.udacity.com/course/intro-to-machine-learning--ud120)

### What is?
- [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Relu](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
- [MNIST - The Hello World of ML](https://en.wikipedia.org/wiki/MNIST_database)
- [Adam optimizer](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

### General notes
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

### Learning
1. [DSA](https://frontendmasters.com/courses/algorithms/)
2. ML Algorithms
    - [X] [Linear regression](https://www.youtube.com/watch?v=VmbA0pi2cRQ)
    - [X] [Support vector machine (SVM)](https://www.youtube.com/watch?v=T9UcK-TxQGw)
    - [X] [Logisitc regression](https://www.youtube.com/watch?v=YYEJ_GUguHw)
    - [X] [K nearest neighbors (KNN)](https://www.youtube.com/watch?v=rTEtEy5o3X0)
    - [ ] [Decision tree](https://www.youtube.com/watch?v=NxEHSAfFlK8&t=5s)
    - [ ] Random forest
    - [X] [Naive Bayes](https://www.youtube.com/watch?v=TLInuAorxqE)
    - [ ] Principal component analysis (PCA)
    - [ ] [Perceptron](https://www.youtube.com/watch?v=aOEoxyA4uXU)
    - [ ] [K means clustering](https://www.youtube.com/watch?v=6UF5Ysk_2gk)
    - Some sort of larger project (maybe bitcoin price predictor)
3. Project: MNIST classifier
4. Learn pytorch or another ml library
5. Neural networks and deep learning
6. RNNs, Transformers and GAN based models

### Ideas I wan to do
- An ML library
    - basic algo's
    - NN's

### Installed libraries
```
python-datasets
python-matplotlib
python-tqdm
python-scikit-learn
python-pandas
python-sympy
python-scipy
```
