from random import randrange

tSet = ([[0, 0], [1, 2], [2, 4], [3, 6], [4, 8]])

def cost(w):
    result = 0.0
    for i in range(len(tSet)):
        x = tSet[i][0]
        y = w*x
        d = y - tSet[i][1]
        result += d*d
    result /= len(tSet)

    return result

def main():
    w = randrange(10)
    h = 1e-3
    rate = 1e-2

    for i in range(500):
        dconst = (cost(w + h) - cost(w)) / h
        w -= rate*dconst
        print("cost:", cost(w), "w:", w)

    print("----------------------")
    for i in range(len(tSet)):
        print("acual:", w*tSet[i][0], "expected:", tSet[i][1])

if __name__ == "__main__":
    main()
