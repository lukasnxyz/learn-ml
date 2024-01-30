import numpy as np
import math

'''
Basics of logs:

solving for x

e ** x = b
'''

if __name__ == "__main__":
    softmax_output = [0.7, 0.1, 0.2] # we want to calc loss on this
    target_output = [1, 0, 0]

    loss = -(math.log(softmax_output[0]) * target_output[0] +
             math.log(softmax_output[1]) * target_output[1] +
             math.log(softmax_output[2]) * target_output[2])

    print(-math.log(softmax_output[0]))
    print(loss)

