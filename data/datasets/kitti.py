import os
import csv
import logging
import random
import pdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from model.heatmap_coder import (
	gaussian_radius,
	draw_umich_gaussian,
	draw_gaussian_1D,
	draw_ellip_gaussian,
	draw_umich_gaussian_2D,
)

from structures.params_3d import ParamsList
from data.augmentations import get_composed_augmentations
from .kitti_utils import Calibration, read_label, approx_proj_center, refresh_attributes, show_heatmap, show_image_with_boxes, show_edge_heatmap

from config import TYPE_ID_CONVERSION

class KITTIDataset(Dataset):
	def __init__(self, cfg, root, is_train=True, transforms=None, augment=True):
		super(KITTIDataset, self).__init__()
		self.root = root
		self.image_dir = os.path.join(root, "image_2")
		self.image_right_dir = os.path.join(root, "image_3")
		self.label_dir = os.path.join(root, "label_2")
		self.calib_dir = os.path.join(root, "calib")

		self.split = cfg.DATASETS.TRAIN_SPLIT if is_train else cfg.DATASETS.TEST_SPLIT
		self.is_train = is_train
		self.transforms = transforms
		self.imageset_txt = os.path.join(root, "ImageSets", "{}.txt".format(self.split))
		assert os.path.exists(self.imageset_txt), "ImageSets file not exist, dir = {}".format(self.imageset_txt)

		image_files = []
		for line in open(self.imageset_txt, "r"):
			base_name = line.replace("\n", "")
			image_name = base_name + ".png"
			image_files.append(image_name)

		self.image_files = image_files
		self.label_files = [i.replace(".png", ".txt") for i in self.image_files]
		self.classes = cfg.DATASETS.DETECT_CLASSES
		self.num_classes = len(self.classes)
		self.num_samples = len(self.image_files)

		# whether to use right-view image
		self.use_right_img = cfg.DATASETS.USE_RIGHT_IMAGE & is_train

		self.augmentation = get_composed_augmentations() if (self.is_train and augment) else None

		# input and output shapes
		self.input_width = cfg.INPUT.WIDTH_TRAIN
		self.input_height = cfg.INPUT.HEIGHT_TRAIN
		self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
		self.output_width = self.input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
		self.output_height = self.input_height // cfg.MODEL.BACKBONE.DOWN_RATIO
		self.output_size = [self.output_width, self.output_height]
		
		# maximal length of extracted feature map when appling edge fusion
		self.max_edge_length = (self.output_width + self.output_height) * 2
		self.max_objs = cfg.DATASETS.MAX_OBJECTS
		
		# filter invalid annotations
		self.filter_annos = cfg.DATASETS.FILTER_ANNO_ENABLE
		self.filter_params = cfg.DATASETS.FILTER_ANNOS

		# handling truncation
		self.consider_outside_objs = cfg.DATASETS.CONSIDER_OUTSIDE_OBJS
		self.use_approx_center = cfg.INPUT.USE_APPROX_CENTER # whether to use approximate representations for outside objects
		self.proj_center_mode = cfg.INPUT.APPROX_3D_CENTER # the type of approximate representations for outside objects

		# for edge feature fusion
		self.enable_edge_fusion = cfg.MODEL.HEAD.ENABLE_EDGE_FUSION

		# True
		self.use_modify_keypoint_visible = cfg.INPUT.KEYPOINT_VISIBLE_MODIFY

		PI = np.pi
		self.orientation_method = cfg.INPUT.ORIENTATION
		self.multibin_size = cfg.INPUT.ORIENTATION_BIN_SIZE
		self.alpha_centers = np.array([0, PI / 2, PI, - PI / 2]) # centers for multi-bin orientation

		# use '2D' or '3D' center for heatmap prediction
		self.heatmap_center = cfg.INPUT.HEATMAP_CENTER
		self.adjust_edge_heatmap = cfg.INPUT.ADJUST_BOUNDARY_HEATMAP # True
		self.edge_heatmap_ratio = cfg.INPUT.HEATMAP_RATIO # radius / 2d box, 0.5

		self.logger = logging.getLogger("monoflex.dataset")
		self.logger.info("Initializing KITTI {} set with {} files loaded.".format(self.split, self.num_samples))

	def __len__(self):
		if self.use_right_img:
			return self.num_samples * 2
		else:
			return self.num_samples

	def get_image(self, idx):
		img_filename = os.path.join(self.image_dir, self.image_files[idx])
		img = Image.open(img_filename).convert('RGB')
		return img

	def get_right_image(self, idx):
		img_filename = os.path.join(self.image_right_dir, self.image_files[idx])
		img = Image.open(img_filename).convert('RGB')
		return img

	def get_calibration(self, idx, use_right_cam=False):
		calib_filename = os.path.join(self.calib_dir, self.label_files[idx])
		return Calibration(calib_filename, use_right_cam=use_right_cam)

	def get_label_objects(self, idx):
		if self.split != 'test':
			label_filename = os.path.join(self.label_dir, self.label_files[idx])
		
		return read_label(label_filename)

	def get_edge_utils(self, image_size, pad_size, down_ratio=4):
		img_w, img_h = image_size

		x_min, y_min = np.ceil(pad_size[0] / down_ratio), np.ceil(pad_size[1] / down_ratio)
		x_max, y_max = (pad_size[0] + img_w - 1) // down_ratio, (pad_size[1] + img_h - 1) // down_ratio

		step = 1
		# boundary idxs
		edge_indices = []
		
		# left
		y = torch.arange(y_min, y_max, step)
		x = torch.ones(len(y)) * x_min
		
		edge_indices_edge = torch.stack((x, y), dim=1)
		edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
		edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
		edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
		edge_indices.append(edge_indices_edge)
		
		# bottom
		x = torch.arange(x_min, x_max, step)
		y = torch.ones(len(x)) * y_max

		edge_indices_edge = torch.stack((x, y), dim=1)
		edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
		edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
		edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
		edge_indices.append(edge_indices_edge)

		# right
		y = torch.arange(y_max, y_min, -step)
		x = torch.ones(len(y)) * x_max

		edge_indices_edge = torch.stack((x, y), dim=1)
		edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
		edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
		edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
		edge_indices.append(edge_indices_edge)

		# top  
		x = torch.arange(x_max, x_min - 1, -step)
		y = torch.ones(len(x)) * y_min

		edge_indices_edge = torch.stack((x, y), dim=1)
		edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
		edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
		edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
		edge_indices.append(edge_indices_edge)

		# concatenate
		edge_indices = torch.cat([index.long() for index in edge_indices], dim=0)

		return edge_indices

	def encode_alpha_multibin(self, alpha, num_bin=2, margin=1 / 6):
		# encode alpha (-PI ~ PI) to 2 classes and 1 regression
		encode_alpha = np.zeros(num_bin * 2)
		bin_size = 2 * np.pi / num_bin # pi
		margin_size = bin_size * margin # pi / 6

		bin_centers = self.alpha_centers
		range_size = bin_size / 2 + margin_size

		offsets = alpha - bin_centers
		offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
		offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi

		for i in range(num_bin):
			offset = offsets[i]
			if abs(offset) < range_size:
				encode_alpha[i] = 1
				encode_alpha[i + num_bin] = offset

		return encode_alpha

	def filtrate_objects(self, obj_list):
		"""
		Discard objects which are not in self.classes (or its similar classes)
		:param obj_list: list
		:return: list
		"""
		type_whitelist = self.classes
		valid_obj_list = []
		for obj in obj_list:
			if obj.type not in type_whitelist:
				continue                
			
			valid_obj_list.append(obj)
		
		return valid_obj_list

	def pad_image(self, image):
		img = np.array(image)
		h, w, c = img.shape
		ret_img = np.zeros((self.input_height, self.input_width, c))
		pad_y = (self.input_height - h) // 2
		pad_x = (self.input_width - w) // 2

		ret_img[pad_y: pad_y + h, pad_x: pad_x + w] = img
		pad_size = np.array([pad_x, pad_y])

		return Image.fromarray(ret_img.astype(np.uint8)), pad_size

	def __getitem__(self, idx):
		
		if idx >= self.num_samples:
			# utilize right color image
			idx = idx % self.num_samples
			img = self.get_right_image(idx)
			calib = self.get_calibration(idx, use_right_cam=True)
			objs = None if self.split == 'test' else self.get_label_objects(idx)

			use_right_img = True
			# generate the bboxes for right color image
			right_objs = []
			img_w, img_h = img.size
			for obj in objs:
				corners_3d = obj.generate_corners3d()
				corners_2d, _ = calib.project_rect_to_image(corners_3d)
				obj.box2d = np.array([max(corners_2d[:, 0].min(), 0), max(corners_2d[:, 1].min(), 0), 
									min(corners_2d[:, 0].max(), img_w - 1), min(corners_2d[:, 1].max(), img_h - 1)], dtype=np.float32)

				obj.xmin, obj.ymin, obj.xmax, obj.ymax = obj.box2d
				right_objs.append(obj)

			objs = right_objs
		else:
			# utilize left color image
			img = self.get_image(idx)
			calib = self.get_calibration(idx)
			objs = None if self.split == 'test' else self.get_label_objects(idx)

			use_right_img = False

		original_idx = self.image_files[idx][:6]
		objs = self.filtrate_objects(objs) # remove objects of irrelevant classes
		 
		# random horizontal flip
		if self.augmentation is not None:
			img, objs, calib = self.augmentation(img, objs, calib)

		# pad image
		img_before_aug_pad = np.array(img).copy()
		img_w, img_h = img.size
		img, pad_size = self.pad_image(img)
		# for training visualize, use the padded images
		ori_img = np.array(img).copy() if self.is_train else img_before_aug_pad

		# the boundaries of the image after padding
		x_min, y_min = int(np.ceil(pad_size[0] / self.down_ratio)), int(np.ceil(pad_size[1] / self.down_ratio))
		x_max, y_max = (pad_size[0] + img_w - 1) // self.down_ratio, (pad_size[1] + img_h - 1) // self.down_ratio

		if self.enable_edge_fusion:
			# generate edge_indices for the edge fusion module
			input_edge_indices = np.zeros([self.max_edge_length, 2], dtype=np.int64)
			edge_indices = self.get_edge_utils((img_w, img_h), pad_size).numpy()
			input_edge_count = edge_indices.shape[0]
			input_edge_indices[:edge_indices.shape[0]] = edge_indices
			input_edge_count = input_edge_count - 1 # explain ? 

		if self.split == 'test':
			# for inference we parametrize with original size
			target = ParamsList(image_size=img.size, is_train=self.is_train)
			target.add_field("pad_size", pad_size)
			target.add_field("calib", calib)
			target.add_field("ori_img", ori_img)
			if self.enable_edge_fusion:
				target.add_field('edge_len', input_edge_count)
				target.add_field('edge_indices', input_edge_indices)

			if self.transforms is not None: img, target = self.transforms(img, target)

			return img, target, original_idx

		# heatmap
		heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
		ellip_heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
		# classification
		cls_ids = np.zeros([self.max_objs], dtype=np.int32) 
		target_centers = np.zeros([self.max_objs, 2], dtype=np.int32)
		# 2d bounding boxes
		gt_bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
		bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
		# keypoints: 2d coordinates and visible(0/1)
		keypoints = np.zeros([self.max_objs, 10, 3], dtype=np.float32)
		keypoints_depth_mask = np.zeros([self.max_objs, 3], dtype=np.float32) # whether the depths computed from three groups of keypoints are valid
		# 3d dimension
		dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
		# 3d location
		locations = np.zeros([self.max_objs, 3], dtype=np.float32)
		# rotation y
		rotys = np.zeros([self.max_objs], dtype=np.float32)
		# alpha (local orientation)
		alphas = np.zeros([self.max_objs], dtype=np.float32)
		# offsets from center to expected_center
		offset_3D = np.zeros([self.max_objs, 2], dtype=np.float32)

		# occlusion and truncation
		occlusions = np.zeros(self.max_objs)
		truncations = np.zeros(self.max_objs)
		
		if self.orientation_method == 'head-axis': orientations = np.zeros([self.max_objs, 3], dtype=np.float32)
		else: orientations = np.zeros([self.max_objs, self.multibin_size * 2], dtype=np.float32) # multi-bin loss: 2 cls + 2 offset

		reg_mask = np.zeros([self.max_objs], dtype=np.uint8) # regression mask
		trunc_mask = np.zeros([self.max_objs], dtype=np.uint8) # outside object mask
		reg_weight = np.zeros([self.max_objs], dtype=np.float32) # regression weight

		for i, obj in enumerate(objs):
			cls = obj.type
			cls_id = TYPE_ID_CONVERSION[cls]
			if cls_id < 0: continue

			# TYPE_ID_CONVERSION = {
			#     'Car': 0,
			#     'Pedestrian': 1,
			#     'Cyclist': 2,
			#     'Van': -4,
			#     'Truck': -4,
			#     'Person_sitting': -2,
			#     'Tram': -99,
			#     'Misc': -99,
			#     'DontCare': -1,
			# }

			float_occlusion = float(obj.occlusion) # 0 for normal, 0.33 for partially, 0.66 for largely, 1 for unknown (mostly very far and small objs)
			float_truncation = obj.truncation # 0 ~ 1 and stands for truncation level

			# bottom centers ==> 3D centers
			locs = obj.t.copy()
			locs[1] = locs[1] - obj.h / 2
			if locs[-1] <= 0: continue # objects which are behind the image

			# generate 8 corners of 3d bbox
			corners_3d = obj.generate_corners3d()
			corners_2d, _ = calib.project_rect_to_image(corners_3d)
			projected_box2d = np.array([corners_2d[:, 0].min(), corners_2d[:, 1].min(), 
										corners_2d[:, 0].max(), corners_2d[:, 1].max()])

			if projected_box2d[0] >= 0 and projected_box2d[1] >= 0 and \
					projected_box2d[2] <= img_w - 1 and projected_box2d[3] <= img_h - 1:
				box2d = projected_box2d.copy()
			else:
				box2d = obj.box2d.copy()

			# filter some unreasonable annotations
			if self.filter_annos:
				if float_truncation >= self.filter_params[0] and (box2d[2:] - box2d[:2]).min() <= self.filter_params[1]: continue

			# project 3d location to the image plane
			proj_center, depth = calib.project_rect_to_image(locs.reshape(-1, 3))
			proj_center = proj_center[0]

			# generate approximate projected center when it is outside the image
			proj_inside_img = (0 <= proj_center[0] <= img_w - 1) & (0 <= proj_center[1] <= img_h - 1)
			
			approx_center = False
			if not proj_inside_img:
				if self.consider_outside_objs:
					approx_center = True

					center_2d = (box2d[:2] + box2d[2:]) / 2
					if self.proj_center_mode == 'intersect':
						target_proj_center, edge_index = approx_proj_center(proj_center, center_2d.reshape(1, 2), (img_w, img_h))
					else:
						raise NotImplementedError
				else:
					continue
			else:
				target_proj_center = proj_center.copy()

			# 10 keypoints
			bot_top_centers = np.stack((corners_3d[:4].mean(axis=0), corners_3d[4:].mean(axis=0)), axis=0)
			keypoints_3D = np.concatenate((corners_3d, bot_top_centers), axis=0)
			keypoints_2D, _ = calib.project_rect_to_image(keypoints_3D)

			# keypoints mask: keypoint must be inside the image and in front of the camera
			keypoints_x_visible = (keypoints_2D[:, 0] >= 0) & (keypoints_2D[:, 0] <= img_w - 1)
			keypoints_y_visible = (keypoints_2D[:, 1] >= 0) & (keypoints_2D[:, 1] <= img_h - 1)
			keypoints_z_visible = (keypoints_3D[:, -1] > 0)

			# xyz visible
			keypoints_visible = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible
			# center, diag-02, diag-13
			keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all()))

			if self.use_modify_keypoint_visible:
				keypoints_visible = np.append(np.tile(keypoints_visible[:4] | keypoints_visible[4:8], 2), np.tile(keypoints_visible[8] | keypoints_visible[9], 2))
				keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(), keypoints_visible[[1, 3, 5, 7]].all()))

				keypoints_visible = keypoints_visible.astype(np.float32)
				keypoints_depth_valid = keypoints_depth_valid.astype(np.float32)

			# downsample bboxes, points to the scale of the extracted feature map (stride = 4)
			keypoints_2D = (keypoints_2D + pad_size.reshape(1, 2)) / self.down_ratio
			target_proj_center = (target_proj_center + pad_size) / self.down_ratio
			proj_center = (proj_center + pad_size) / self.down_ratio
			
			box2d[0::2] += pad_size[0]
			box2d[1::2] += pad_size[1]
			box2d /= self.down_ratio
			# 2d bbox center and size
			bbox_center = (box2d[:2] + box2d[2:]) / 2
			bbox_dim = box2d[2:] - box2d[:2]

			# target_center: the point to represent the object in the downsampled feature map
			if self.heatmap_center == '2D':
				target_center = bbox_center.round().astype(np.int)
			else:
				target_center = target_proj_center.round().astype(np.int)

			# clip to the boundary
			target_center[0] = np.clip(target_center[0], x_min, x_max)
			target_center[1] = np.clip(target_center[1], y_min, y_max)

			pred_2D = True # In fact, there are some wrong annotations where the target center is outside the box2d
			if not (target_center[0] >= box2d[0] and target_center[1] >= box2d[1] and target_center[0] <= box2d[2] and target_center[1] <= box2d[3]):
				pred_2D = False

			if (bbox_dim > 0).all() and (0 <= target_center[0] <= self.output_width - 1) and (0 <= target_center[1] <= self.output_height - 1):
				rot_y = obj.ry
				alpha = obj.alpha

				# generating heatmap
				if self.adjust_edge_heatmap and approx_center:
					# for outside objects, generate 1-dimensional heatmap
					bbox_width = min(target_center[0] - box2d[0], box2d[2] - target_center[0])
					bbox_height = min(target_center[1] - box2d[1], box2d[3] - target_center[1])
					radius_x, radius_y = bbox_width * self.edge_heatmap_ratio, bbox_height * self.edge_heatmap_ratio
					radius_x, radius_y = max(0, int(radius_x)), max(0, int(radius_y))
					assert min(radius_x, radius_y) == 0
					heat_map[cls_id] = draw_umich_gaussian_2D(heat_map[cls_id], target_center, radius_x, radius_y)						
				else:
					# for inside objects, generate circular heatmap
					radius = gaussian_radius(bbox_dim[1], bbox_dim[0])
					radius = max(0, int(radius))
					heat_map[cls_id] = draw_umich_gaussian(heat_map[cls_id], target_center, radius)

				cls_ids[i] = cls_id
				target_centers[i] = target_center
				# offset due to quantization for inside objects or offset from the interesection to the projected 3D center for outside objects
				offset_3D[i] = proj_center - target_center
				
				# 2D bboxes
				gt_bboxes[i] = obj.box2d.copy() # for visualization
				if pred_2D: bboxes[i] = box2d

				# local coordinates for keypoints
				keypoints[i] = np.concatenate((keypoints_2D - target_center.reshape(1, -1), keypoints_visible[:, np.newaxis]), axis=1)
				keypoints_depth_mask[i] = keypoints_depth_valid
				
				dimensions[i] = np.array([obj.l, obj.h, obj.w])
				locations[i] = locs
				rotys[i] = rot_y
				alphas[i] = alpha

				orientations[i] = self.encode_alpha_multibin(alpha, num_bin=self.multibin_size)

				reg_mask[i] = 1
				reg_weight[i] = 1 # all objects are of the same weights (for now)
				trunc_mask[i] = int(approx_center) # whether the center is truncated and therefore approximate
				occlusions[i] = float_occlusion
				truncations[i] = float_truncation

		# visualization
		# img3 = show_image_with_boxes(img, cls_ids, target_centers, bboxes.copy(), keypoints, reg_mask, 
		# 							offset_3D, self.down_ratio, pad_size, orientations, vis=True)
		# show_heatmap(img, heat_map, index=original_idx)

		target = ParamsList(image_size=img.size, is_train=self.is_train) 
		target.add_field("cls_ids", cls_ids)
		target.add_field("target_centers", target_centers)
		target.add_field("keypoints", keypoints)
		target.add_field("keypoints_depth_mask", keypoints_depth_mask)
		target.add_field("dimensions", dimensions)
		target.add_field("locations", locations)
		target.add_field("calib", calib)
		target.add_field("reg_mask", reg_mask)
		target.add_field("reg_weight", reg_weight)
		target.add_field("offset_3D", offset_3D)
		target.add_field("2d_bboxes", bboxes)
		target.add_field("pad_size", pad_size)
		target.add_field("ori_img", ori_img)
		target.add_field("rotys", rotys)
		target.add_field("trunc_mask", trunc_mask)
		target.add_field("alphas", alphas)
		target.add_field("orientations", orientations)
		target.add_field("hm", heat_map)
		target.add_field("gt_bboxes", gt_bboxes) # for validation visualization
		target.add_field("occlusions", occlusions)
		target.add_field("truncations", truncations)

		if self.enable_edge_fusion:
			target.add_field('edge_len', input_edge_count)
			target.add_field('edge_indices', input_edge_indices)

		if self.transforms is not None: img, target = self.transforms(img, target)

		return img, target, original_idx
