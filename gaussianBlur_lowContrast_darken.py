import cv2

img = cv2.imread("chang_im_input/71040.jpg")

img = cv2.GaussianBlur(img, (7,7), 0)

img = cv2.convertScaleAbs(
    img,
    alpha=0.75,
    beta=-15
)

cv2.imwrite("im8.jpg", img)