import cv2
import numpy as np

img = cv2.imread("chang_im_input/18891.jpg")

h, w = img.shape[:2]

M = cv2.getRotationMatrix2D((w//2, h//2), 5, 1)

img = cv2.warpAffine(img, M, (w, h))

noise = np.random.normal(0, 12, img.shape).astype(np.int16)

img = np.clip(
    img.astype(np.int16) + noise,
    0,
    255
).astype(np.uint8)

cv2.imwrite("im6.jpg", img)