import cv2

img = cv2.imread("chang_im_input/5135.jpg")

# Downsample then upsample
small = cv2.resize(img, None, fx=0.5, fy=0.5)
img = cv2.resize(small, (img.shape[1], img.shape[0]))

# JPEG compression
cv2.imwrite(
    "temp.jpg",
    img,
    [cv2.IMWRITE_JPEG_QUALITY, 40]
)

result = cv2.imread("temp.jpg")

cv2.imwrite("im4.jpg", result)