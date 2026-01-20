## Stanford lecture notes

### Lecture 3
- Batch gradient descent:
    - Is limited as it has to go through whole dataset for each iteration of training
- Stochastic gradient descent:
    - Used more for much larger data as it doesn't have to go through each point in each epoch
    - Randomly chooses a point and optimizes from it's output
    - Slowly decrease learning rate over time so that the oscillation further into training becomes
      smaller and smaller
    - Parameters oscillate towards the end of the training (reaching the lowest error)
