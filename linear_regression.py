import pandas as pd
import matplotlib.pyplot as plt

def loss_function(m, b, points):
    # necesarry as we need the partial derivative of the mse for gradient decent
    # this is the mean squared error function
    # tells us by how much we are off from the actual result

    # this only works well if the data is inherently linear, that's why a curve like x^2
    #   is a terrible test

    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (m * x + b)) ** 2

    total_error /= float(len(points))

    return total_error

def gradient_decent(m_now, b_now, points, rate): # rate is learning rate
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    # using the partial derivatives of mean squared error function with respect
    #   to m and b

    for i in range(n):
        x = float(points.iloc[i].x)
        y = float(points.iloc[i].y)

        m_gradient += -(2/n) * x * (y - ((m_now * x) + b_now))
        b_gradient += -(2/n) * (y - ((m_now * x) + b_now))

    m = m_now - rate * m_gradient
    b = b_now - rate * b_gradient

    # m = m_now - m_gradient * rate
    # b = b_now - b_gradient * rate

    return m, b

def main():
    # data = pd.read_csv("data/height-weight-no-label.csv")
    data = pd.read_csv("data/times_two.csv")
    # just looks like this: [[0, 0], [1, 2], [2, 4], [3, 6]] until x = 10

    m = 0
    b = 0
    rate = 0.0001
    epochs = 1000

    for i in range(epochs):
        m, b = gradient_decent(m, b, data, rate)
        print(f"Epoch: {i}")

    loss = loss_function(m, b, data)
    print(m, b) # any number
    print(loss) # want close to 0
    print(f"Test x = 40, y = 80, yhat: {m * 40 + b}")

    plt.scatter(data.x, data.y, color="black")
    #plt.plot(list(range(55, 80)), [m * x + b for x in range(55, 80)], color="red")
    plt.plot(list(range(0, 10)), [m * x + b for x in range(0, 10)], color="red")
    plt.title(str(loss))
    plt.show()

if __name__ == "__main__":
    main()
