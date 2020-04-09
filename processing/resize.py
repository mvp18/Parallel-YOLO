import cv2
import glob

images = glob.glob("*.png")

i = 1
for x in images:

    img = cv2.imread(x, -1)

    re = cv2.resize(img, (416, 416))

    cv2.imwrite("resized_" + str(i) + ".png", re)
    i = i + 1
