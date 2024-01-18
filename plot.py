from matplotlib import pyplot as plt

def square(x):
    return x * x

def deriv(func, x):
    h = 1e-9
    return (func(x + h) - func(x))/h

def main():
    x = []
    y = []
    bot = -10
    top = 11

    _, axs = plt.subplots(2)

    for i in range(bot, top):
        x.append(i)
        y.append(square(i))

    axs[0].plot(x, y)
    axs[0].set_title("x^2")

    x.clear()
    y.clear()

    for i in range(bot, top):
        x.append(i)
        y.append(deriv(square, i))

    axs[1].plot(x, y)
    axs[1].set_title("d/dx x^2")

    plt.show()

if __name__ == "__main__":
    main()
