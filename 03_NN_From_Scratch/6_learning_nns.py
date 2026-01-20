import numpy as np

if __name__ == "__main__":
    layer_outputs = [[4.8, 1.21, 2.385],
                    [8.9, -1.81, 0.2],
                    [1.41, 1.051, 0.026]]

    exp_values = np.exp(layer_outputs)
    # we want to sum by all the rows and keep a vactor output

    norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    # axis 0 is sum of columns
    # axis 1 is sum of rows
    # keepdims makes it a proper 3,1 vector instead of an array

    print(norm_values)
