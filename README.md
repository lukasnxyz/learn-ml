## learn-ml
This is a basic learning plan I'm working on to get myself up to speed on the basics of machine learning and
understanding the fields ML, DL and other related topics for becoming an ML/Software Engineer. This
guide will be a work in progress as I am learning along side finding the proper resources to learn
from.

#### Other tips and motivation
Self study is very difficult so having a set plan like one would have in a school system will help a lot.
[Here](https://machinelearningmastery.com/machine-learning-for-programmers/) you can find a good guide on
machine learning for software developers. It goes more for the top down approach of learning instead of
bottom up, but an interesting read nonetheless.
[An interesting article on how to learn on your own](https://metacademy.org/roadmaps/rgrosse/learn_on_your_own)

#### Bottom up method
You can jump around within each step. I would advise to just skim the Stanford lectures if you understand
how the previous topics work from implementing them.
1. DSA
    - [X] Most used data structures
    - [X] Most used algorithms
2. Fundamental ML algorithms from scratch
    - [X] Linear regression
        - [Video tutorial](https://www.youtube.com/watch?v=VmbA0pi2cRQ)
    - [X] Logistic regression
        - [Video tutorial](https://www.youtube.com/watch?v=YYEJ_GUguHw)
    - [X] Support vector machine (SVM)
        - [Video tutorial](https://www.youtube.com/watch?v=T9UcK-TxQGw)
        - [Article tutorial](https://towardsdatascience.com/implement-multiclass-svm-from-scratch-in-python-b141e43dc084)
        - [The kernel trick](https://www.youtube.com/watch?v=Q7vT0--5VII)
    - [X] K nearest neighbors (KNN)
        - [Video tutorial](https://www.youtube.com/watch?v=rTEtEy5o3X0)
    - [X] Naive Bayes
        - [Video tutorial](https://www.youtube.com/watch?v=TLInuAorxqE)
    - [ ] Decision tree
        - [Video tutorial](https://www.youtube.com/watch?v=NxEHSAfFlK8&t=5s)
        - [Video explanation](https://www.youtube.com/watch?v=LDRbO9a6XPU)
    - [ ] Random forest
    - [X] Principal component analysis (PCA)
        - [Video tutorial](https://www.youtube.com/watch?v=Rjr62b_h7S4)
    - [X] K means clustering
        - [Video tutorial](https://www.youtube.com/watch?v=6UF5Ysk_2gk)
    - [X] Perceptron
        - [Video tutorial](https://www.youtube.com/watch?v=aOEoxyA4uXU)
        - [A Little About Perceptrons and Activation Functions](https://medium.com/mlearning-ai/a-little-about-perceptrons-and-activation-functions-aed19d672656)
3. Learn basics of pytorch (or any other popular ML library)
    - [X] [learnpytorch.io](https://www.learnpytorch.io/)
4. Projects:
    - [X] Random algo implement from scratch with math notes
    - [X] MNIST classifier (using pytorch)
5. Multilayer Perceptron/Feed Forward neural network
    - [X] [Here's](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3) a good intro to NN's playlist. Warning: it's not a complete guide!
    - [Training a classifier with softmax cross entropy loss](https://douglasorr.github.io/2021-10-training-objectives/1-xent/article.html#mjx-eqn-eqn%3Aloss)
    - [X] [NN from scratch Karpathy (basically implement working MLP](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
6. Karpathy Zero-To-Hero Series
    - [X] [Micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
    - [X] [Makemore: pt1](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2)
    - [X] [Makemore: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)
    - [X] [Makemore: Activations and Gradients](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4)
    - [X] [Makemore: Backprop](https://www.youtube.com/watch?v=q8SA3rM6ckI&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5)
    - [X] [Makemore: WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=6)
    - [X] [GPT From Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7)
7. Andrew Ng - The Deep Learning Specialization
    - [X] Neural Network and Deep Learning
    - [X] Improving Deep Neural Networks
    - [X] Structuring Machine Learning Projects
    - [X] Convolutional Neural Networks
    - [X] Sequence Models
8. CUDA/GPU Programming and Parallel Computing
    - [ ] [Time-lapse guy from X](https://www.youtube.com/watch?v=86FAWCzIe_4&t=2263s)
    - [ ] [Github repo for it](https://github.com/Infatoshi/cuda-course)
        - [X] 01 Deep Learning Ecosystem
        - [ ] 02 Setup
        - [ ] 03 C and C++ Review
        - [ ] 04 Gentle Intro to GPUS
        - [ ] 05 Writing your first kernels
        - [ ] 06 CUDA APIs
        - [ ] 07 Faster Matmul
        - [ ] 08 Triton
        - [ ] 09 Pytorch extensions
        - [ ] 10 Final Project
        - [ ] 11 Extras
9. Compilers, LLVM, and such (low level things for deep learning)
    - [ ] [Tutorials on Tinygrad](https://mesozoic-egg.github.io/tinygrad-notes/)
    - [ ] [Language Front-end with LLVM](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)

#### Datasets I've used in this repo
- [500 Person Gender-Height-Weight-BMI](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex)
- [MNIST](https://yann.lecun.com/exdb/mnist/), [MNIST Wikipedia](https://en.wikipedia.org/wiki/MNIST_database)
- [Height weight](https://www.kaggle.com/code/martandsay/height-weight-regression-classification/input)
- [Water quality](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- [Diabetes](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
