import matplotlib.pyplot as plt
from pandas import read_csv

def main():
    data = read_csv("../data/height-weight.csv")
    plt.scatter(data.height, data.weight)

    plt.show()

if __name__ == "__main__":
    main()
