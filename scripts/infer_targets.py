import os
import cv2
import numpy as np
from config import *
import sys
import glob

base_path = os.getcwd() + "/"
eval_path = str(sys.argv[1])
mode = str(sys.argv[2])

def get_classes():
	obj_dict = {}
	obj_id = 0
	for filename in os.listdir(base_path + '../data/' + mode + '/Annotations/'):
		with open(base_path + '../data/' + mode + '/Annotations/' + filename, 'r') as f:
			data = f.read()
		
		lines = data.split('\n')
		
		num_objects = int(lines[0])
		lines = lines[1:]

		for i_ in range(num_objects):
			line = lines[i_].split(' ')
			label = line[0]

			if not label in obj_dict.keys():
				obj_dict[label] = obj_id
				obj_id += 1

	num_objects = len(obj_dict.keys())
	idx2obj = {value:key for key,value in obj_dict.items()}

	return num_objects, obj_dict, idx2obj

num_classes, _, _ = get_classes()
filenames = glob.glob(eval_path + "/*.txt")

for filename in filenames:
	print(filename)
	with open(filename, 'r') as f:
		data = f.read()

	data = data.split('\n')

	arr = np.array([float(d) for d in data if len(d) > 1])
	# print(arr)
	# print(np.unique(arr))

	targets = np.reshape(arr, (batch_size,  num_anchors, 5 + num_classes, feature_map_size[0], feature_map_size[0]))

	predictions = []

	downscale_factor = H/feature_map_size[0]
	
	for i in range(feature_map_size[0]):
		for j in range(feature_map_size[0]):
			for ai in range(num_anchors):
				confidence = targets[0,ai,0,i,j]
				if confidence < CONF_THRESH:
					continue

				pred_x = targets[0,ai,1,i,j]
				pred_y = targets[0,ai,2,i,j]
				pred_h = targets[0,ai,3,i,j]
				pred_w = targets[0,ai,4,i,j]

				center_x = (pred_x + j)*downscale_factor
				center_y = (pred_y + i)*downscale_factor

				h = anchors[ai][0] * (pred_h/(1.0 - pred_h + 1e-4))
				w = anchors[ai][1] * (pred_w/(1.0 - pred_w + 1e-4))

				pred_class = np.argmax(targets[0,ai,5:,i,j])

				# print(targets[0,ai,5:,i,j])
				
				predictions.append([confidence, center_x, center_y, h, w, pred_class])

	print(predictions)
