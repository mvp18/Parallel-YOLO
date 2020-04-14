import os
import cv2
import numpy as np
from config import *
import sys
import glob

filename = "mask_resized_1.txt"
with open(filename, 'r') as f:
	data = f.read()

data = data.split('\n')
arr = np.array([float(d) for d in data if len(d) > 1])

print(np.unique(arr))
print(np.sum(arr))
