### Notes
- Saving models is just saving it's weights and biases
- In ML we us gpu's over cpu's because they have way more cores for calculations
- Usually try to keep all weights/biases between -1 and 1
- Parts of a NN:
    -
- Each neuron has an activation function
    - Why use activation function? Without one, all the neuron outputs are always linear. If you fit
      a non linear function with it, it's not possible. Activation function can help fit non linear
      functions.
    - Output layers usually have different ones than input layers
    - Why use mostly relu? Because it's super fast and simply works
    - Relu is most used: tweak activation point with bias and slope of relu by tweaking weight
- Is it all almost like taylor polynomial thing?
- Activation functions continued:
    - Most activation functions only look at single neurons and cannot not regularize across all
      neurons
    - Bounding issue
- Softmax:
    - Input -> Exponentiate -> Normalize -> Output
    - Softmax is: Exponentiate -> Normalize
    - Softmaxt is used mainly in the last layer of a nn to predict the class
- General loss function: Catagorical Cross Entropy
    - Taking negative sum of target value * log of predicted value for each value in the
      distribution
    - Simplified to -log of predicted target class because of one-shot vectors
    - **Loss** is measurment of error, the further away an output from the target class, the higher
      the loss (error)

