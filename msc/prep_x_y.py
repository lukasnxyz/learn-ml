from taylor import MathFunc

def main():
    mf = MathFunc() # for func1

    max = 100
    x = []
    y = []

    for i in range(max):
        x.append(i)
        y.append(mf.square(i))

    with open('func_data.csv', 'w') as file:
        file.write("x,y\n")
        for i in range(max):
            line = str(x[i]) + "," + str(y[i]) + "\n"
            file.write(line)

if __name__ == "__main__":
    main()
