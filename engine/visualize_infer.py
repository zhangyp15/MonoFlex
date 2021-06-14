import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
cv2.setNumThreads(0)
import os
import pdb

from PIL import Image

from scipy.optimize import minimize
from config import TYPE_ID_CONVERSION
from shapely.geometry import Polygon
from config import cfg
from utils.visualizer import Visualizer
from data.datasets.kitti_utils import draw_projected_box3d, draw_box3d_on_top, init_bev_image, draw_bev_box3d

keypoint_colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], 
				[190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], 
				[107, 142, 35], [255, 0, 0], [0, 0, 142], [0, 0, 70],
				[152, 251, 152], [0, 130, 180], [220, 20, 60], [0, 60, 100]]

def box_iou(box1, box2):
	intersection = max((min(box1[2], box2[2]) - max(box1[0], box2[0])), 0) * max((min(box1[3], box2[3]) - max(box1[1], box2[1])), 0)
	union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

	return intersection / union

def box_iou_3d(corner1, corner2):
	# for height overlap, since y face down, use the negative y
	min_h_a = -corner1[0:4, 1].sum() / 4.0
	max_h_a = -corner1[4:8, 1].sum() / 4.0
	min_h_b = -corner2[0:4, 1].sum() / 4.0
	max_h_b = -corner2[4:8, 1].sum() / 4.0

	# overlap in height
	h_max_of_min = max(min_h_a, min_h_b)
	h_min_of_max = min(max_h_a, max_h_b)
	h_overlap = max(0, h_min_of_max - h_max_of_min)

	if h_overlap == 0:
		return 0

	# x-z plane overlap
	box1, box2 = corner1[0:4, [0, 2]], corner2[0:4, [0, 2]]
	bottom_a, bottom_b = Polygon(box1), Polygon(box2)
	if bottom_a.is_valid and bottom_b.is_valid:
		# check is valid, A valid Polygon may not possess any overlapping exterior or interior rings.
		bottom_overlap = bottom_a.intersection(bottom_b).area

	overlap_3d = bottom_overlap * h_overlap        
	union3d = bottom_a.area * (max_h_a - min_h_a) + bottom_b.area * (max_h_b - min_h_b) - overlap_3d

	return overlap_3d / union3d

def box3d_to_corners(locs, dims, roty):
	# 3d bbox template
	h, w, l = dims
	x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
	y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
	z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

	# rotation matirx
	R = np.array([[np.cos(roty), 0, np.sin(roty)],
				  [0, 1, 0],
				  [-np.sin(roty), 0, np.cos(roty)]])

	corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
	corners3d = np.dot(R, corners3d).T
	corners3d = corners3d + locs

	return corners3d


# visualize for test-set
def show_image_with_boxes_test(image, output, target, visualize_preds):
	# output Tensor:
	# clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores
	image = image.numpy().astype(np.uint8)
	output = output.cpu().float().numpy()
	
	# filter results with visualization threshold
	vis_thresh = cfg.TEST.VISUALIZE_THRESHOLD
	output = output[output[:, -1] > vis_thresh]

	ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}
	clses = output[:, 0]
	box2d = output[:, 2:6]
	dims = output[:, 6:9]
	locs = output[:, 9:12]
	rotys = output[:, 12]
	score = output[:, 13]
	keypoints = visualize_preds['keypoints'].cpu()
	proj_center = visualize_preds['proj_center'].cpu()

	calib = target.get_field('calib')
	pad_size = target.get_field('pad_size')

	# B x C x H x W  ----> H x W x C
	pred_heatmap = visualize_preds['heat_map']
	all_heatmap = np.asarray(pred_heatmap[0, :, ...].cpu().sum(dim=0))
	all_heatmap = cv2.resize(all_heatmap, (1280, 384))
	all_heatmap = all_heatmap[pad_size[1] : pad_size[1] + image.shape[0], pad_size[0] : pad_size[0] + image.shape[1]]

	img2 = Visualizer(image.copy()) # for 2d bbox
	img3 = image.copy() # for 3d bbox
	img4 = init_bev_image() # for bev
	img_keypoint = image.copy() 
	font = cv2.FONT_HERSHEY_SIMPLEX

	pred_color = (0, 255, 0)
	# plot prediction 
	for i in range(box2d.shape[0]):
		img2.draw_box(box_coord=box2d[i], edge_color='g')
		img2.draw_text(text='{}, {:.3f}'.format(ID_TYPE_CONVERSION[clses[i]], score[i]), position=(int(box2d[i, 0]), int(box2d[i, 1])))

		corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
		corners_2d, depth = calib.project_rect_to_image(corners3d)
		img3 = draw_projected_box3d(img3, corners_2d, color=pred_color)

		corners3d_lidar = calib.project_rect_to_velo(corners3d)
		img4 = draw_box3d_on_top(img4, corners3d_lidar[np.newaxis, :], thickness=2, color=pred_color, scores=None)
		# 10 x 2
		keypoint_i = (keypoints[i].view(-1, 2) + proj_center[i].view(-1, 2)) * 4 - pad_size.view(1, 2)
		# depth from keypoint
		center_height = keypoint_i[-2, -1] - keypoint_i[-1, -1]
		edge_height = keypoint_i[:4, -1] - keypoint_i[4:8, -1]
		# depth of four edges
		edge_depth = calib.f_u * dims[i, 0] / edge_height
		center_depth = calib.f_u * dims[i, 0] / center_height
		edge_depth = [edge_depth[[0, 3]].mean(), edge_depth[[1, 2]].mean()]
		# print(locs[i, -1], center_depth, edge_depth)

		for i in range(keypoint_i.shape[0]):
			cv2.circle(img_keypoint, tuple(keypoint_i[i]), 4, keypoint_colors[i], -1)

	img2 = img2.output.get_image()
	heat_mixed = img2.astype(np.float32) / 255 + all_heatmap[..., np.newaxis] * np.array([1, 0, 0]).reshape(1, 1, 3)
	img3 = img3.astype(np.float32) / 255
	stacked_img = np.vstack((heat_mixed, img3))

	plt.figure()
	plt.imshow(stacked_img)
	plt.title('2D and 3D results')
	plt.show()

# heatmap and 3D detections
def show_image_with_boxes(image, output, target, visualize_preds, vis_scores=None):
	# output Tensor:
	# clses, pred_alphas, pred_box2d, pred_dimensions, pred_locations, pred_rotys, scores
	image = image.numpy().astype(np.uint8)
	output = output.cpu().float().numpy()

	if vis_scores is not None:
		output[:, -1] = vis_scores.squeeze().cpu().float().numpy()
	
	# filter results with visualization threshold
	vis_thresh = cfg.TEST.VISUALIZE_THRESHOLD
	output = output[output[:, -1] > vis_thresh]
	ID_TYPE_CONVERSION = {k : v for v, k in TYPE_ID_CONVERSION.items()}

	# predictions
	clses = output[:, 0]
	box2d = output[:, 2:6]
	dims = output[:, 6:9]
	locs = output[:, 9:12]
	rotys = output[:, 12]
	score = output[:, 13]

	proj_center = visualize_preds['proj_center'].cpu()
	keypoints = visualize_preds['keypoints'].cpu()

	# ground-truth
	calib = target.get_field('calib')
	pad_size = target.get_field('pad_size')
	valid_mask = target.get_field('reg_mask').bool()
	trunc_mask = target.get_field('trunc_mask').bool()
	num_gt = valid_mask.sum()
	gt_clses = target.get_field('cls_ids')[valid_mask]
	gt_boxes = target.get_field('gt_bboxes')[valid_mask]
	gt_locs = target.get_field('locations')[valid_mask]
	gt_dims = target.get_field('dimensions')[valid_mask]
	gt_rotys = target.get_field('rotys')[valid_mask]

	print('detections / gt objs: {} / {}'.format(box2d.shape[0], num_gt))

	pred_heatmap = visualize_preds['heat_map']
	all_heatmap = np.asarray(pred_heatmap[0, 0, ...].cpu())
	all_heatmap = cv2.resize(all_heatmap, (image.shape[1], image.shape[0]))

	img2 = Visualizer(image.copy()) # for 2d bbox
	img3 = image.copy() # for 3d bbox
	img4 = init_bev_image() # for bev

	font = cv2.FONT_HERSHEY_SIMPLEX
	pred_color = (0, 255, 0)
	gt_color = (255, 0, 0)

	# plot prediction 
	for i in range(box2d.shape[0]):
		img2.draw_box(box_coord=box2d[i], edge_color='g')
		img2.draw_text(text='{}, {:.3f}'.format(ID_TYPE_CONVERSION[clses[i]], score[i]), position=(int(box2d[i, 0]), int(box2d[i, 1])))

		corners3d = box3d_to_corners(locs[i], dims[i], rotys[i])
		corners_2d, depth = calib.project_rect_to_image(corners3d)
		img3 = draw_projected_box3d(img3, corners_2d, cls=ID_TYPE_CONVERSION[clses[i]], color=pred_color, draw_corner=False)

		corners3d_lidar = calib.project_rect_to_velo(corners3d)
		img4 = draw_bev_box3d(img4, corners3d[np.newaxis, :], thickness=2, color=pred_color, scores=None)

	# plot ground-truth
	for i in range(num_gt):
		img2.draw_box(box_coord=gt_boxes[i], edge_color='r')

		# 3d bbox template
		l, h, w = gt_dims[i]
		x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
		y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
		z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

		# rotation matirx
		roty = gt_rotys[i]
		R = np.array([[np.cos(roty), 0, np.sin(roty)],
					  [0, 1, 0],
					  [-np.sin(roty), 0, np.cos(roty)]])

		corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
		corners3d = np.dot(R, corners3d).T
		corners3d = corners3d + gt_locs[i].numpy() + np.array([0, h / 2, 0]).reshape(1, 3)

		corners_2d, depth = calib.project_rect_to_image(corners3d)
		img3 = draw_projected_box3d(img3, corners_2d, color=gt_color, draw_corner=False)

		corners3d_lidar = calib.project_rect_to_velo(corners3d)
		img4 = draw_bev_box3d(img4, corners3d[np.newaxis, :], thickness=2, color=gt_color, scores=None)

	img2 = img2.output.get_image()
	heat_mixed = img2.astype(np.float32) / 255 + all_heatmap[..., np.newaxis] * np.array([1, 0, 0]).reshape(1, 1, 3)
	img4 = cv2.resize(img4, (img3.shape[0], img3.shape[0]))
	stack_img = np.concatenate([img3, img4], axis=1)

	plt.figure(figsize=(12, 8))
	plt.subplot(211)
	plt.imshow(all_heatmap); plt.title('heatmap'); plt.axis('off')
	plt.subplot(212)
	plt.imshow(stack_img); plt.title('2D/3D boxes'); plt.axis('off')
	plt.suptitle('Detections')
	plt.show()