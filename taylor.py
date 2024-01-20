from matplotlib import pyplot as plt
from math import factorial, sqrt
import sympy

class MathFunc():
    def square(self, x):
        return x ** 2

    def cube(self, x):
        return x ** 3

    def func1(self, x):
        # from my recent calc exam lol
        return (x ** 3) - (6 * (x ** 2)) + (9 * x)

    def sqrootx(self, x):
        return sqrt(x)

class TaylorSeries():
    def __init__(self, f, count):
        self.f = f
        self.count = count
        self.coefficients = []

        self.__find_coefficients()

    def __find_coefficients(self):
        for i in range(0, self.count + 1): # i is which degree of deriv here
            x = sympy.symbols("x")
            deriv = sympy.diff(self.f, x, i)
            print(deriv)
            # self.coefficients.append(deriv / factorial(i))

    def print_equation(self):
        print(self.coefficients)

    def approx_value(self, x):
        fx = 0
        for i in range(len(self.coefficients)):
            fx += self.coefficients[i] * (x ** i) # coefficient to the *nth term (** power op)

        return fx

    def get_func(self):
        return self.f

def main():
    x = []
    y = []
    bot = -2
    top = 3

    mf = MathFunc()

    ts = TaylorSeries(mf.cube, 2)
    ts.print_equation()

    for i in range(bot, top):
        x.append(i)
        y.append(ts.f(i))

    plt.plot(x, y, label=(ts.get_func().__name__))

    x.clear()
    y.clear()

    for i in range(bot, top):
        x.append(i)
        y.append(ts.approx_value(i))

    plt.plot(x, y, label=("taylor: " + ts.get_func().__name__))

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
