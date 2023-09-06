import cv2 as opencv

def capture_image():
    cam = opencv.VideoCapture(0)

    result, image = cam.read()

    if result:
        opencv.imwrite("captured_image.jpg", image)

    else:
        print("No image detected!")

def detect_image():
    img = opencv.imread("captured_image.jpg")

    img_gray = opencv.cvtColor(img, opencv.COLOR_BGR2GRAY)
    img_rgb = opencv.cvtColor(img, opencv.COLOR_BGR2RGB)

    try:
        stop_data = opencv.CascadeClassifier('stop_data.xml')
    except opencv.error as e:
        print(f"OpenCV Error: {e}")
    except Exception as ex:
        print(f"An error occurred: {ex}")

    found = stop_data.detectMultiScale(img_gray, minSize = (20, 20))

    # Don't do anything if there's
    # no sign
    amount_found = len(found)

    if amount_found != 0:

        # There may be more than one
        # sign in the image
        for (x, y, width, height) in found:

            # We draw a green rectangle around
            # every recognized sign
            opencv.rectangle(img_rgb, (x, y), (x + height, y + width), (0, 255, 0), 5)

    # Creates the environment of
    # the picture and shows it
    plt.subplot(1, 1, 1)
    plt.imshow(img_rgb)
    plt.show()

def main():
    capture_image()
    detect_image()

if __name__ == "__main__":
    main()
