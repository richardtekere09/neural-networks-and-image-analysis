import cv2
import numpy as np

# Read Image
img = cv2.imread(
    "/Users/richard/neural-networks-and-image-analysis/Photoshop-filter/input/image.jpg"
)
if img is None:
    raise FileNotFoundError("Could not read the image. Check the path.")


def nothing(x):  # trackbar callback (required)
    pass


def brightness(img):
    win = "Brightness"
    cv2.namedWindow(win)  # window must exist before creating the trackbar
    cv2.createTrackbar("val", win, 100, 150, nothing)  # 0..150 -> scale 0..1.5

    while True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)

        val = cv2.getTrackbarPos("val", win) / 100.0  # 0.0–1.5

        # S channel (index 1)
        hsv[:, :, 1] *= val
        # V channel (index 2)
        hsv[:, :, 2] *= val

        # Clip and convert back
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        img_bright = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        cv2.imshow(win, img_bright)

        # Quit on 'q'
        if (cv2.waitKey(30) & 0xFF) == ord("q"):
            break

        # Also exit if the user clicks the window's X button
        # NOTE: use the SAME window name as above
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def tv_60(img):
    cv2.namedWindow("image")
    cv2.createTrackbar("val", "image", 0, 255, nothing)
    cv2.createTrackbar("threshold", "image", 0, 100, nothing)
    while True:
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.getTrackbarPos("threshold", "image")
        val = cv2.getTrackbarPos("val", "image")
        for i in range(height):
            for j in range(width):
                if np.random.randint(100) <= thresh:
                    if np.random.randint(2) == 0:
                        gray[i, j] = min(
                            gray[i, j] + np.random.randint(0, val + 1), 255
                        )  # adding noise to image and setting values > 255 to 255.
                    else:
                        gray[i, j] = max(
                            gray[i, j] - np.random.randint(0, val + 1), 0
                        )  # subtracting noise to image and setting values < 0 to 0.
        #cv2.imshow("Original", img)
        cv2.imshow("image", gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

def exponential_function(channel, exp):
    table = np.array([min((i**exp), 255) for i in np.arange(0, 256)]).astype(
        "uint8"
    )  # generating table for exponential function
    channel = cv2.LUT(channel, table)
    return channel


def duo_tone(img):
    cv2.namedWindow("image")
    cv2.createTrackbar("exponent", "image", 0, 10, nothing)
    switch1 = "0 : BLUE n1 : GREEN n2 : RED"
    cv2.createTrackbar(switch1, "image", 1, 2, nothing)
    switch2 = "0 : BLUE n1 : GREEN n2 : RED n3 : NONE"
    cv2.createTrackbar(switch2, "image", 3, 3, nothing)
    switch3 = "0 : DARK n1 : LIGHT"
    cv2.createTrackbar(switch3, "image", 0, 1, nothing)
    while True:
        exp = cv2.getTrackbarPos("exponent", "image")
        exp = 1 + exp / 100  # converting exponent to range 1-2
        s1 = cv2.getTrackbarPos(switch1, "image")
        s2 = cv2.getTrackbarPos(switch2, "image")
        s3 = cv2.getTrackbarPos(switch3, "image")
        res = img.copy()
        for i in range(3):
            if i in (s1, s2):  # if channel is present
                res[:, :, i] = exponential_function(
                    res[:, :, i], exp
                )  # increasing the values if channel selected
            else:
                if s3:  # for light
                    res[:, :, i] = exponential_function(
                        res[:, :, i], 2 - exp
                    )  # reducing value to make the channels light
                else:  # for dark
                    res[:, :, i] = 0  # converting the whole channel to 0
       # cv2.imshow("Original", img)
        cv2.imshow("image", res)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

def sepia(img):
    res = img.copy()
    res = cv2.cvtColor(
        res, cv2.COLOR_BGR2RGB
    )  # converting to RGB as sepia matrix is for RGB
    res = np.array(res, dtype=np.float64)
    res = cv2.transform(
        res,
        np.matrix(
            [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
        ),
    )
    res[np.where(res > 255)] = 255  # clipping values greater than 255 to 255
    res = np.array(res, dtype=np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imshow("original", img)
    cv2.imshow("Sepia", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Calling Functions
brightness(img)
#tv_60(img)
#duo_tone(img)
#sepia(img)