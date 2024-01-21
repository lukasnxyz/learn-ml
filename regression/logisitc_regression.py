import matplotlib.pyplot as plt
from pandas import read_csv
from numpy import exp

def sigmoid(x):
    return 1/(1 + exp(-x))

def gradient_descent(m_now, b_now, points, rate):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = float(points.iloc[i].x)
        y = float(points.iloc[i].y)

        linear_pred = m_now * x + b_now
        pred = sigmoid(linear_pred)

        m_gradient = (1/n) * (x * (pred - y))
        b_gradient = (1/n) * (pred - y)

    m = m_now - m_gradient * rate
    b = b_now - b_gradient * rate

    return m, b

def train(m, b, points, rate, epochs):
    for i in range(epochs):
        m, b = gradient_descent(m, b, points, rate)
        print(f"{i}: m = {m}, b = {b}")

    return m, b

def predict(m, b, x):
    linear_pred = m * x + b
    pred = sigmoid(linear_pred)

    if pred <= 0.5:
        return 0
    else:
        return 1

def main():
    # this is all wrong!
    data = read_csv("../data/height-weight.csv")

    # need to use logistic regression for true or false problems
    # this is more of a true or false algo because of the sigmoid function
    # more for categorical data and classification

    m = 0
    b = 0
    epochs = 250
    rate = 0.001

    m, b = train(m, b, data, rate, epochs)

    plt.scatter(data.x, data.y, color="black")
    plt.plot(list(range(62, 75)), [m * x + b for x in range(62, 75)], color="red")
    plt.show()

if __name__ == "__main__":
    main()
