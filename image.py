from matplotlib.image import imread

def main():
    image = imread("bochum1.jpg")
    print(image.shape)

if __name__ == "__main__":
    main()
