import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def least_squares_reg(x: np.ndarray, y: np.ndarray) -> list:
    N = len(x) if len(x) == len(y) else 0
    m = (N*sum(x*y) - sum(x)*sum(x)) / (N*sum(x**2) - sum(x)**2)
    b = (sum(y) - m*sum(x)) / N
    return m, b

def l_of_b_fit(x: np.ndarray, y: np.ndarray) -> list:
    m = (y[-1]-y[0])/(x[-1]-x[0])
    b = y[0]-m*x[0]
    return m, b

def corr_coef(x: np.ndarray, y: np.ndarray) -> float: 
    xm, ym = sum(x)/len(x), sum(y)/len(y)
    return sum((x-xm)*(y-ym))/sqrt(sum((x-xm)**2)*sum((y-ym)**2))

def plot_points(x: np.ndarray, y: np.ndarray, t: str, f):
    yb_pos = list(filter(lambda x : x > 0, f(x)))
    x_pos = list(filter(lambda x : f(x) > 0, x))
    plt.scatter(x, y), plt.plot(x_pos, yb_pos)
    plt.xlabel("x"), plt.ylabel("y"), plt.title(t)
    plt.show()

if __name__ == "__main__":
    x = np.array([6, 12, 13, 17, 22, 25, 27, 29, 30, 32])
    y = np.array([45, 47, 39, 58, 68, 76, 75, 74, 78, 81])
    print(f"x: min:{min(x)}, max:{max(x)}\ny: min:{min(y)}, max:{max(y)}")

    m, b = l_of_b_fit(x, y)
    print(f"y = {m:.3f}x + {b:.3f}")
    r = corr_coef(x, y)
    f = lambda x : m*x + b
    plot_points(x, y, f"r = {r:.3f}", f)
