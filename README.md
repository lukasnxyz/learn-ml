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
Know what [MNIST](https://en.wikipedia.org/wiki/MNIST_database) is.

#### Bottom up method
You can jump around within each step. I would advise to just skim the Stanford lectures if you understand
how the previous topics work from implementing them.
1. DSA
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
    - [ ] Principal component analysis (PCA)
        - [Video tutorial](https://www.youtube.com/watch?v=Rjr62b_h7S4)
    - [X] K means clustering
        - [Video tutorial](https://www.youtube.com/watch?v=6UF5Ysk_2gk)
    - [X] Perceptron
        - [Video tutorial](https://www.youtube.com/watch?v=aOEoxyA4uXU)
        - [A Little About Perceptrons and Activation Functions](https://medium.com/mlearning-ai/a-little-about-perceptrons-and-activation-functions-aed19d672656)
3. Stanford CS229 ML course on youtube until neural networks
    - [X] [Lecture 1 - 10](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=1)
4. Read [The 100 Page Machine Learning Book](https://themlbook.com/) until the neural networks chapter
5. Learn basics of pytorch (or any other popular ML library)
6. Projects:
    - [ ] Random algo implement from scratch with math notes
    - [X] MNIST classifier (using pytorch)
7. Multilayer Perceptron/Feed Forward neural network
    - [X] [Here's](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3) a good intro to NN's playlist. Warning: it's not a complete guide!
    - [Training a classifier with softmax cross entropy loss](https://douglasorr.github.io/2021-10-training-objectives/1-xent/article.html#mjx-eqn-eqn%3Aloss)
    - [X] [NN from scratch Karpathy (basically implement working MLP](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
8. Convolutional and Recurrent neural networks
    - [ ] [Karpathy on nn's at stanford CS231n](https://www.youtube.com/watch?v=i94OvYb6noo)
    - [ ] [Included practice to CS231n](https://cs231n.github.io/)
9. Stanford CS229 ML course on youtube neural networks until end
    - [ ] [Lecture 11-20](https://www.youtube.com/watch?v=MfIjxPh6Pys&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=11)
10. Finish The 100 Page Machine Learning Book from step 4
11. Projects:
    - [ ] MNIST classifier from scratch
    - [ ] A language model?
12. Transformers

#### Other notes and useful links
- [Python Machine Learning Mini-Course](https://machinelearningmastery.com/python-machine-learning-mini-course/)
- [Basics of numpy](https://numpy.org/devdocs/user/absolute_beginners.html)
- [Folder structure for an ML project](https://dev.to/luxacademy/generic-folder-structure-for-your-machine-learning-projects-4coe)
- [MNIST from scratch George Hotz](https://www.youtube.com/watch?v=JRlyw6LO5qo&list=WL&index=1)
- [Building a Language Model](https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d)
- [Neural Network from Scratch](https://medium.com/@waleedmousa975/building-a-neural-network-from-scratch-using-numpy-and-math-libraries-a-step-by-step-tutorial-in-608090c20466)
- [MNIST classifier using Pytorch (my own article)](https://lukasn.xyz/posts/neural-network-model-basics/)

### Download the datasets I've used in this repo
- [500 Person Gender-Height-Weight-BMI](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex)
- [MNIST](https://www.kaggle.com/datasets/avnishnish/mnist-original)
- [Height weight](https://www.kaggle.com/code/martandsay/height-weight-regression-classification/input)
- [Water quality](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- [Diabetes](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- [Other data](https://github.com/Avik-Jain/100-Days-Of-ML-Code/tree/master/datasets)
