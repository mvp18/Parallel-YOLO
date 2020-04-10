import cv2
import glob
import os

def make_dirs():
	if not os.path.exists(os.getcwd() + "/../data/"):
		os.mkdir(os.getcwd() + "/../data/")

	if not os.path.exists(os.getcwd() + "/../data/standard/images"):
		os.makedirs(os.getcwd() + "/../data/standard/images")

make_dirs()

images = glob.glob(os.getcwd() + "/../sample/images/*.png")
i = 1
for x in images:
    img = cv2.imread(x, -1)
    re = cv2.resize(img, (416, 416))
    cv2.imwrite(os.getcwd() + "/../data/standard/images/resized_" + str(i) + ".png", re)
    i = i + 1
