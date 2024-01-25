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
- [Project structure](https://dev.to/luxacademy/generic-folder-structure-for-your-machine-learning-projects-4coe)

### Tutorials
- [Stanford CS229 ML course](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
- [Stanford coursea ml course](https://www.coursera.org/learn/machine-learning)
- [NNs from zero to hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [NN from scatch in python/numpy](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- [MNIST classifier from scratch article](https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab)
- [MNIST classifier from scratch](https://github.com/kdexd/digit-classifier)
- [Need to know algos](https://gab41.lab41.org/the-10-algorithms-machine-learning-engineers-need-to-know-f4bb63f5b2fa#.ofc7t2965)
- [Python Machine Learning Mini-Course](https://machinelearningmastery.com/python-machine-learning-mini-course/)

### Other Resources
- [ML for programmers](https://machinelearningmastery.com/machine-learning-for-programmers/)
- [How to learn on your own](https://metacademy.org/roadmaps/rgrosse/learn_on_your_own)
- [The 100 Page Machine Learning Book](https://themlbook.com/)

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

### Learning
1. [DSA](https://frontendmasters.com/courses/algorithms/)
2. ML Algorithms
    - [X] [Linear regression](https://www.youtube.com/watch?v=VmbA0pi2cRQ)
    - [X] [Support vector machine (SVM)](https://www.youtube.com/watch?v=T9UcK-TxQGw)
        - https://towardsdatascience.com/implement-multiclass-svm-from-scratch-in-python-b141e43dc084
        - https://www.youtube.com/watch?v=Q7vT0--5VII
        - https://stackoverflow.com/questions/62242579/implementing-svm-rbf
    - [X] [Logisitc regression](https://www.youtube.com/watch?v=YYEJ_GUguHw)
    - [X] [K nearest neighbors (KNN)](https://www.youtube.com/watch?v=rTEtEy5o3X0)
    - [ ] [Naive Bayes](https://www.youtube.com/watch?v=TLInuAorxqE)
    - [ ] [Decision tree](https://www.youtube.com/watch?v=NxEHSAfFlK8&t=5s)
    - [ ] Random forest
    - [ ] Principal component analysis (PCA)
    - [ ] [K means clustering](https://www.youtube.com/watch?v=6UF5Ysk_2gk)
    - [ ] [Perceptron](https://www.youtube.com/watch?v=aOEoxyA4uXU)
3. Stanford CS229 ML course on youtube until neural networks
4. Read The 100 ML Book until neural networks
5. Project: MNIST classifier
6. Learn basics of pytorch
7. Perceptron and Feed Forward neural networks
8. Convolutional and Recurrent neural networks
9. Transformers

### Ideas I want to build
- An ML library
    - basic algo's
    - NN's
- Chatbot (language model)
