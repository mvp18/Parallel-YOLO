import os
import cv2
import numpy as np
from config import *
import sys

base_path = os.getcwd() + "/"
mode = str(sys.argv[1])

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

def bb_intersection_over_union(boxA, boxB):
	"""
	box : [xmin, ymin, xmax, ymax]
	"""
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def create_data(num_classes, obj2idx, idx2obj):
	if not os.path.exists(base_path + '../data/' + mode + '/targets'):
		os.mkdir(base_path + '../data/' + mode + '/targets')

	obj_dict = {} # To store object labels
	for filename in os.listdir(base_path + '../data/' + mode + '/Annotations/'):
		with open(base_path + '../data/' + mode + '/Annotations/' + filename, 'r') as f:
			data = f.read()
		
		lines = data.split('\n')
		
		num_objects = int(lines[0])
		lines = lines[1:]

		target_dict = {}
		for i_ in range(num_objects):
			line = lines[i_].split(' ')
			label = line[0]
			xmin = float(line[1]) - 1 # For indexing
			ymin = float(line[2]) - 1 # For indexing
			xmax = float(line[3]) - 1 # For indexing
			ymax = float(line[4]) - 1 # For indexing

			matching_anchor = {}
			iou_max = -1

			for fw in feature_map_size:
				downscale_factor = H/fw

				for grid_x in range(fw):
					for grid_y in range(fw):
						gt_xmin = xmin/downscale_factor
						gt_ymin = ymin/downscale_factor
						gt_xmax = xmax/downscale_factor
						gt_ymax = ymax/downscale_factor

						x_center = (gt_xmin + gt_xmax)/2
						y_center = (gt_ymin + gt_ymax)/2

						if not (x_center >= grid_x and x_center < grid_x+1 and y_center >= grid_y and y_center < grid_y+1):
							continue

						for ai, anchor in enumerate(anchors):
							anchor_h = anchor[0]/downscale_factor
							anchor_w = anchor[1]/downscale_factor

							anchor_xmin = grid_x + 0.5 - anchor_w/2
							anchor_xmax = grid_x + 0.5 + anchor_w/2
							anchor_ymin = grid_y + 0.5 - anchor_h/2
							anchor_ymax = grid_y + 0.5 + anchor_h/2

							gt_box = [gt_xmin,gt_ymin,gt_xmax,gt_ymax]
							anchor_box = [anchor_xmin,anchor_ymin,anchor_xmax,anchor_ymax]
							iou = bb_intersection_over_union(gt_box,anchor_box)

							if iou > iou_max:
								matching_anchor['fw'] = fw
								matching_anchor['grid_x'] = grid_x
								matching_anchor['grid_y'] = grid_y
								matching_anchor['anchor'] = anchor
								matching_anchor['anchor_id'] = ai

			target_dict[i_] = {}
			target_dict[i_]['label'] = label
			target_dict[i_]['fw'] = matching_anchor['fw']
			target_dict[i_]['grid_x'] = matching_anchor['grid_x']
			target_dict[i_]['grid_y'] = matching_anchor['grid_y']
			target_dict[i_]['anchor'] = matching_anchor['anchor']
			target_dict[i_]['anchor_id'] = matching_anchor['anchor_id']
			target_dict[i_]['gt_x'] = (xmax + xmin)/(2 * downscale_factor)
			target_dict[i_]['gt_y'] = (ymax + ymin)/(2 * downscale_factor)
			target_dict[i_]['gt_w'] = (xmax - xmin)/(downscale_factor)
			target_dict[i_]['gt_h'] = (ymax - ymin)/(downscale_factor)

		targets = np.zeros((batch_size, feature_map_size[0], feature_map_size[0], num_anchors, 5 + num_classes))
		masks = np.zeros((batch_size, feature_map_size[0], feature_map_size[0], num_anchors, 5 + num_classes))

		# anchor2idx = {anchors[i]:i for i in range(num_anchors)}
		masks[:, :, :, :, 0] = 1 # To account for confidence predictions

		for i in range(num_objects):
			x = target_dict[i]['grid_x']
			y = target_dict[i]['grid_y']
			gt_x = target_dict[i]['gt_x']
			gt_y = target_dict[i]['gt_y']
			gt_w = target_dict[i]['gt_w']
			gt_h = target_dict[i]['gt_h']
			anchor_id = target_dict[i]['anchor_id']
			label_id = obj2idx[target_dict[i]['label']]

			downscale_factor = H/feature_map_size[0]
			anchor_h = anchor[0]/downscale_factor
			anchor_w = anchor[1]/downscale_factor

			target_x = gt_x - x
			target_y = gt_y - y
			target_h = gt_h/(gt_h + anchor_h)
			target_w = gt_w/(gt_w + anchor_w)

			masks[0, y, x, anchor_id, :] = 1
			targets[0, y, x, anchor_id, :5] = [1.0, target_x, target_y, target_h, target_w]
			targets[0, y, x, anchor_id, 5 + label_id] = 1

		# masks and targets need to be written to a file
		np.savetxt(base_path + '../data/' + mode + '/targets' + '/mask_' + filename, masks.ravel())
		np.savetxt(base_path + '../data/' + mode + '/targets' + '/target_' + filename, targets.ravel())

num_classes, obj2idx, idx2obj = get_classes()
create_data(num_classes, obj2idx, idx2obj)
