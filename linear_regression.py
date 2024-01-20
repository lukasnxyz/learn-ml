import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint

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

    m = m_now - m_gradient * rate
    b = b_now - b_gradient * rate

    return m, b

def train(m, b, data, rate, epochs):
    bar = tqdm(range(epochs))
    bar.set_description("Training")
    for _ in bar:
        #bar.set_description("Training. " + "m: {:.5f}, ".format(m) + "b: {:.5f}".format(b))
        m, b = gradient_decent(m, b, data, rate)

    return m, b

def main():
    data = pd.read_csv("data/height-weight-no-label.csv")

    m = randint(0, 10) # I think it's pretty important to pick a proper range for weights
    b = randint(100, 200) * -1
    rate = 1e-4
    epochs = 1000

    # Getting increased loss values instead of converging to a minimum is usually a sign that learning rate is too high.
    # https://stackoverflow.com/questions/39314946/why-does-my-linear-regression-get-nan-values-instead-of-learning

    m, b = train(m, b, data, rate, epochs)

    loss = loss_function(m, b, data)
    print("m:", m, "b:", b) # any number
    print("MSE:", loss) # want close to 0 although 0 is not possible for a not fully linear function

    plt.scatter(data.x, data.y, color="black")
    plt.plot(list(range(62, 75)), [m * x + b for x in range(62, 75)], color="red")
    plt.title(str(loss))
    plt.show()

if __name__ == "__main__":
    main()
