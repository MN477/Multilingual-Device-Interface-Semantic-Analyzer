import cv2
import numpy as np

img = cv2.imread("chang_im_input/48836.jpg")

kernel = np.zeros((3,3))
kernel[1,:] = np.ones(3)
kernel /= 3

img = cv2.filter2D(img, -1, kernel)

small = cv2.resize(img, None, fx=0.6, fy=0.6)
img = cv2.resize(small, (img.shape[1], img.shape[0]))

cv2.imwrite(
    "temp.jpg",
    img,
    [cv2.IMWRITE_JPEG_QUALITY, 35]
)

result = cv2.imread("temp.jpg")

cv2.imwrite("im7.jpg", result)