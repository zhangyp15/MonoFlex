import numpy as np
import pdb
import torch
import torch.nn.functional as F

import torchvision.ops.roi_align as roi_align
from data.datasets.kitti_utils import convertAlpha2Rot
PI = np.pi

class Anno_Encoder():
		def __init__(self, cfg):
			device = cfg.MODEL.DEVICE
			self.INF = 100000000
			self.EPS = 1e-3

			# center related
			self.num_cls = len(cfg.DATASETS.DETECT_CLASSES)
			self.min_radius = cfg.DATASETS.MIN_RADIUS
			self.max_radius = cfg.DATASETS.MAX_RADIUS
			self.center_ratio = cfg.DATASETS.CENTER_RADIUS_RATIO
			self.target_center_mode = cfg.INPUT.HEATMAP_CENTER
			# if mode == 'max', centerness is the larger value, if mode == 'area', assigned to the smaller bbox
			self.center_mode = cfg.MODEL.HEAD.CENTER_MODE
			
			# depth related
			self.depth_mode = cfg.MODEL.HEAD.DEPTH_MODE
			self.depth_range = cfg.MODEL.HEAD.DEPTH_RANGE
			self.depth_ref = torch.as_tensor(cfg.MODEL.HEAD.DEPTH_REFERENCE).to(device=device)

			# dimension related
			self.dim_mean = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_MEAN).to(device=device)
			self.dim_std = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_STD).to(device=device)
			self.dim_modes = cfg.MODEL.HEAD.DIMENSION_REG

			# orientation related
			self.alpha_centers = torch.tensor([0, PI / 2, PI, - PI / 2]).to(device=device)
			self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
			self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE

			# offset related
			self.offset_mean = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[0]
			self.offset_std = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[1]

			# output info
			self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
			self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
			self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
			self.K = self.output_width * self.output_height

		@staticmethod
		def rad_to_matrix(rotys, N):
			device = rotys.device

			cos, sin = rotys.cos(), rotys.sin()

			i_temp = torch.tensor([[1, 0, 1],
								 [0, 1, 0],
								 [-1, 0, 1]]).to(dtype=torch.float32, device=device)

			ry = i_temp.repeat(N, 1).view(N, -1, 3)

			ry[:, 0, 0] *= cos
			ry[:, 0, 2] *= sin
			ry[:, 2, 0] *= sin
			ry[:, 2, 2] *= cos

			return ry

		def decode_box2d_fcos(self, centers, pred_offset, pad_size=None, out_size=None):
			box2d_center = centers.view(-1, 2)
			box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
			# left, top, right, bottom
			box2d[:, :2] = box2d_center - pred_offset[:, :2]
			box2d[:, 2:] = box2d_center + pred_offset[:, 2:]

			# for inference
			if pad_size is not None:
				N = box2d.shape[0]
				out_size = out_size[0]
				# upscale and subtract the padding
				box2d = box2d * self.down_ratio - pad_size.repeat(1, 2)
				# clamp to the image bound
				box2d[:, 0::2].clamp_(min=0, max=out_size[0].item() - 1)
				box2d[:, 1::2].clamp_(min=0, max=out_size[1].item() - 1)

			return box2d

		def encode_box3d(self, rotys, dims, locs):
			'''
			construct 3d bounding box for each object.
			Args:
					rotys: rotation in shape N
					dims: dimensions of objects
					locs: locations of objects

			Returns:

			'''
			if len(rotys.shape) == 2:
					rotys = rotys.flatten()
			if len(dims.shape) == 3:
					dims = dims.view(-1, 3)
			if len(locs.shape) == 3:
					locs = locs.view(-1, 3)

			device = rotys.device
			N = rotys.shape[0]
			ry = self.rad_to_matrix(rotys, N)

			# l, h, w
			dims_corners = dims.view(-1, 1).repeat(1, 8)
			dims_corners = dims_corners * 0.5
			dims_corners[:, 4:] = -dims_corners[:, 4:]
			index = torch.tensor([[4, 5, 0, 1, 6, 7, 2, 3],
								[0, 1, 2, 3, 4, 5, 6, 7],
								[4, 0, 1, 5, 6, 2, 3, 7]]).repeat(N, 1).to(device=device)
			
			box_3d_object = torch.gather(dims_corners, 1, index)
			box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1))
			box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

			return box_3d.permute(0, 2, 1)

		def decode_depth(self, depths_offset):
			'''
			Transform depth offset to depth
			'''
			if self.depth_mode == 'exp':
				depth = depths_offset.exp()
			elif self.depth_mode == 'linear':
				depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
			elif self.depth_mode == 'inv_sigmoid':
				depth = 1 / torch.sigmoid(depths_offset) - 1
			else:
				raise ValueError

			if self.depth_range is not None:
				depth = torch.clamp(depth, min=self.depth_range[0], max=self.depth_range[1])

			return depth

		def decode_location_flatten(self, points, offsets, depths, calibs, pad_size, batch_idxs):
			batch_size = len(calibs)
			gts = torch.unique(batch_idxs, sorted=True).tolist()
			locations = points.new_zeros(points.shape[0], 3).float()
			points = (points + offsets) * self.down_ratio - pad_size[batch_idxs]

			for idx, gt in enumerate(gts):
				corr_pts_idx = torch.nonzero(batch_idxs == gt).squeeze(-1)
				calib = calibs[gt]
				# concatenate uv with depth
				corr_pts_depth = torch.cat((points[corr_pts_idx], depths[corr_pts_idx, None]), dim=1)
				locations[corr_pts_idx] = calib.project_image_to_rect(corr_pts_depth)

			return locations

		def decode_depth_from_keypoints(self, pred_offsets, pred_keypoints, pred_dimensions, calibs, avg_center=False):
			# pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
			assert len(calibs) == 1 # for inference, batch size is always 1
			
			calib = calibs[0]
			# we only need the values of y
			pred_height_3D = pred_dimensions[:, 1]
			pred_keypoints = pred_keypoints.view(-1, 10, 2)
			# center height -> depth
			if avg_center:
				updated_pred_keypoints = pred_keypoints - pred_offsets.view(-1, 1, 2)
				center_height = updated_pred_keypoints[:, -2:, 1]
				center_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (center_height.abs() * self.down_ratio * 2)
				center_depth = center_depth.mean(dim=1)
			else:
				center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
				center_depth = calib.f_u * pred_height_3D / (center_height.abs() * self.down_ratio)
			
			# corner height -> depth
			corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
			corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
			corner_02_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (corner_02_height * self.down_ratio)
			corner_13_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (corner_13_height * self.down_ratio)
			corner_02_depth = corner_02_depth.mean(dim=1)
			corner_13_depth = corner_13_depth.mean(dim=1)
			# K x 3
			pred_depths = torch.stack((center_depth, corner_02_depth, corner_13_depth), dim=1)

			return pred_depths

		def decode_depth_from_keypoints_batch(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
			# pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
			pred_height_3D = pred_dimensions[:, 1].clone()
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = pred_dimensions.new_zeros(pred_dimensions.shape[0])

			center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
			corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
			corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]

			pred_keypoint_depths = {'center': [], 'corner_02': [], 'corner_13': []}

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):			
				calib = calibs[idx]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				center_depth = calib.f_u * pred_height_3D[corr_pts_idx] / (F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
				corner_02_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
				corner_13_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)

				corner_02_depth = corner_02_depth.mean(dim=1)
				corner_13_depth = corner_13_depth.mean(dim=1)

				pred_keypoint_depths['center'].append(center_depth)
				pred_keypoint_depths['corner_02'].append(corner_02_depth)
				pred_keypoint_depths['corner_13'].append(corner_13_depth)

			for key, depths in pred_keypoint_depths.items():
				pred_keypoint_depths[key] = torch.clamp(torch.cat(depths), min=self.depth_range[0], max=self.depth_range[1])

			pred_depths = torch.stack([depth for depth in pred_keypoint_depths.values()], dim=1)

			return pred_depths

		def decode_dimension(self, cls_id, dims_offset):
			'''
			retrieve object dimensions
			Args:
					cls_id: each object id
					dims_offset: dimension offsets, shape = (N, 3)

			Returns:

			'''
			cls_id = cls_id.flatten().long()
			cls_dimension_mean = self.dim_mean[cls_id, :]

			if self.dim_modes[0] == 'exp':
				dims_offset = dims_offset.exp()

			if self.dim_modes[2]:
				cls_dimension_std = self.dim_std[cls_id, :]
				dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
			else:
				dimensions = dims_offset * cls_dimension_mean
				
			return dimensions

		def decode_axes_orientation(self, vector_ori, locations):
			'''
			retrieve object orientation
			Args:
					vector_ori: local orientation in [axis_cls, head_cls, sin, cos] format
					locations: object location

			Returns: for training we only need roty
							 for testing we need both alpha and roty

			'''
			if self.multibin:
				pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
				pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
				orientations = vector_ori.new_zeros(vector_ori.shape[0])
				for i in range(self.orien_bin_size):
					mask_i = (pred_bin_cls.argmax(dim=1) == i)
					s = self.orien_bin_size * 2 + i * 2
					e = s + 2
					pred_bin_offset = vector_ori[mask_i, s : e]
					orientations[mask_i] = torch.atan2(pred_bin_offset[:, 0], pred_bin_offset[:, 1]) + self.alpha_centers[i]
			else:
				axis_cls = torch.softmax(vector_ori[:, :2], dim=1)
				axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
				head_cls = torch.softmax(vector_ori[:, 2:4], dim=1)
				head_cls = head_cls[:, 0] < head_cls[:, 1]
				# cls axis
				orientations = self.alpha_centers[axis_cls + head_cls * 2]
				sin_cos_offset = F.normalize(vector_ori[:, 4:])
				orientations += torch.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])

			locations = locations.view(-1, 3)
			rays = torch.atan2(locations[:, 0], locations[:, 2])
			alphas = orientations
			rotys = alphas + rays

			larger_idx = (rotys > PI).nonzero()
			small_idx = (rotys < -PI).nonzero()
			if len(larger_idx) != 0:
					rotys[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					rotys[small_idx] += 2 * PI

			larger_idx = (alphas > PI).nonzero()
			small_idx = (alphas < -PI).nonzero()
			if len(larger_idx) != 0:
					alphas[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					alphas[small_idx] += 2 * PI

			return rotys, alphas

if __name__ == '__main__':
	pass