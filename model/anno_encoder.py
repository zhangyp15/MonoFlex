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

			self.depth_refine_bin_num = cfg.MODEL.DEPTH_REFINE.BIN_NUM
			self.depth_refine_bin_size = cfg.MODEL.DEPTH_REFINE.BIN_SIZE
			self.depth_bin_offset = torch.arange(- self.depth_refine_bin_num / 2, self.depth_refine_bin_num / 2).ceil().to(device) * self.depth_refine_bin_size

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

			# roi align size
			self.roi_size = cfg.MODEL.DEPTH_REFINE.OUTPUT_SIZE
			self.patch_jitter = cfg.MODEL.DEPTH_REFINE.JITTER
			self.refine_depth_detach = cfg.MODEL.DEPTH_REFINE.DETACH_DEPTH

			# uncertainty
			self.uncertainty_range = cfg.MODEL.HEAD.UNCERTAINTY_RANGE

			# init all locations in the feature map
			h, w = self.output_height, self.output_width
			shifts_x = torch.arange(0, w, dtype=torch.float32)
			shifts_y = torch.arange(0, h, dtype=torch.float32)
			ys, xs = torch.meshgrid(shifts_y, shifts_x)
			self.uv_map = torch.stack((xs, ys))

			xs = xs.reshape(-1)
			ys = ys.reshape(-1)
			self.xs = xs
			self.ys = ys

		def encode_patch(self, pred_corners, batch_idxs, calibs, pad_size, pred_depths=None, compute_xyz_roi=True):
			batch_size = len(calibs)
			pred_patchs = []

			for i in range(batch_size):
				calib = calibs[i]
				################
				## pred patch ##
				################
				pred_corners_i = pred_corners[torch.where(batch_idxs == i)].reshape(-1, 3)
				pred_corners_2D_i, pred_corners_depth_i = calib.project_rect_to_image_tensor(pred_corners_i)
				pred_corners_2D_i = pred_corners_2D_i.view(-1, 8, 2)
				# corners to patch
				pred_patch_i = torch.stack((pred_corners_2D_i[..., 0].min(dim=1)[0], pred_corners_2D_i[..., 1].min(dim=1)[0], 
										pred_corners_2D_i[..., 0].max(dim=1)[0], pred_corners_2D_i[..., 1].max(dim=1)[0]), dim=1)

				pred_patch_i = (pred_patch_i + pad_size[i].view(1, 2).repeat(1, 2)) / self.down_ratio
				# float -> int and clamp
				pred_patch_i = pred_patch_i.ceil()

				pred_patch_i[:, [0, 1]].clamp_(min=0)
				pred_patch_i[:, 2].clamp_(max=self.output_width)
				pred_patch_i[:, 3].clamp_(max=self.output_height)
				pred_patchs.append(pred_patch_i)

			pred_patchs = torch.cat(pred_patchs, dim=0)
			pred_patchs = torch.cat((batch_idxs[:, None].float(), pred_patchs), dim=1)

			pred_xyz_rois = []
			if compute_xyz_roi:
				# uv + depth
				pred_uvz = torch.cat((self.uv_map.type_as(pred_depths)[None].repeat(batch_size, 1, 1, 1), pred_depths), dim=1)
				# roi align
				pred_uvz_rois = roi_align(pred_uvz, pred_patchs, self.roi_size)
				for i in range(batch_size):
					pred_uvz_roi = pred_uvz_rois[batch_idxs == i]
					# transform u, v to original coordinates
					pred_uvz_roi[:, :2] = pred_uvz_roi[:, :2] * self.down_ratio - pad_size[i].view(1, -1, 1, 1)
					# image uv -> rect 
					pred_xyz_roi = calib.project_image_to_rect(pred_uvz_roi.permute(0, 2, 3, 1).reshape(-1, 3))
					pred_xyz_roi = pred_xyz_roi.reshape(pred_uvz_roi.shape[0], self.roi_size[0], self.roi_size[1], 3).permute(0, 3, 1, 2)
					pred_xyz_rois.append(pred_xyz_roi)
			
				pred_xyz_rois = torch.cat(pred_xyz_rois, dim=0)

			return pred_patchs, pred_xyz_rois

		def refine_depth(self, pred_depth, pred_refine):
			pred_refine_clses = pred_refine[:, :self.depth_refine_bin_num]
			pred_refine_offset = pred_refine[:, self.depth_refine_bin_num]
			# choose the best bin 
			pred_bin_idxs = pred_refine_clses.argmax(dim=1)
			# add offset to coarse depth
			pred_refined_depth = pred_depth + pred_refine_offset + self.depth_bin_offset[pred_bin_idxs]

			return pred_refined_depth

		def encode_box2d(self, K, rotys, dims, locs, img_size):
			device = rotys.device
			K = K.to(device=device)

			img_size = img_size.flatten()

			box3d = self.encode_box3d(rotys, dims, locs)
			box3d_image = torch.matmul(K, box3d)
			box3d_image = box3d_image[:, :2, :] / box3d_image[:, 2, :].view(
					box3d.shape[0], 1, box3d.shape[2]
			)

			xmins, _ = box3d_image[:, 0, :].min(dim=1)
			xmaxs, _ = box3d_image[:, 0, :].max(dim=1)
			ymins, _ = box3d_image[:, 1, :].min(dim=1)
			ymaxs, _ = box3d_image[:, 1, :].max(dim=1)

			xmins = xmins.clamp(0, img_size[0])
			xmaxs = xmaxs.clamp(0, img_size[0])
			ymins = ymins.clamp(0, img_size[1])
			ymaxs = ymaxs.clamp(0, img_size[1])

			bboxfrom3d = torch.cat((xmins.unsqueeze(1), ymins.unsqueeze(1),
															xmaxs.unsqueeze(1), ymaxs.unsqueeze(1)), dim=1)

			return bboxfrom3d

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

		def sample_region(self, target_centers, gt, xs, ys):
			# sample regions
			num_gt = gt.shape[0]

			# work for both 2D and 3D center, but less accurate for 2D due to quantification
			center_x = target_centers[:, 0]
			center_y = target_centers[:, 1]

			# only works for 2D centers
			# centers = (gt[:, :2] + gt[:, 2:]) / 2
			# center_x = centers[:, 0]
			# center_y = centers[:, 1]

			center_gt = gt.new_zeros(gt.shape)
			x_radius = ((gt[..., 2] - gt[..., 0]) * self.center_ratio).clamp_(min=self.min_radius, max=self.max_radius)
			y_radius = ((gt[..., 3] - gt[..., 1]) * self.center_ratio).clamp_(min=self.min_radius, max=self.max_radius)
			
			# no gt
			if center_x.sum() > 0:
				# decrease the size of bboxes
				xmin, xmax = center_x - x_radius, center_x + x_radius
				ymin, ymax = center_y - y_radius, center_y + y_radius
				center_gt[:, 0] = torch.max(gt[:, 0], xmin)
				center_gt[:, 1] = torch.max(gt[:, 1], ymin)
				center_gt[:, 2] = torch.min(gt[:, 2], xmax)
				center_gt[:, 3] = torch.min(gt[:, 3], ymax)

				l = xs[:, None] - center_gt[:, 0][None]
				t = ys[:, None] - center_gt[:, 1][None]
				r = center_gt[:, 2][None] - xs[:, None]
				b = center_gt[:, 3][None] - ys[:, None]

				edge_dis = torch.stack([l, t, r, b], dim=2)
				try:
					inside_mask = edge_dis.min(dim=2)[0] >= 0
				except:
					pdb.set_trace()
			else:
				inside_mask = torch.zeros(xs.shape[0], num_gt).to(gt.device)

			return inside_mask

		def decode_offset(self, offset_3D, regression_2D):
			offset_3D = offset_3D * self.offset_std + self.offset_mean
			dimensions_2D = (regression_2D[:, :2] + regression_2D[:, 2:4]) / 2
			offset_3D = offset_3D *  dimensions_2D

			return offset_3D

		def encode_centerness(self, target_centers, batch_bboxes, clses):
			xs = self.xs.clone().type_as(batch_bboxes)
			ys = self.ys.clone().type_as(batch_bboxes)
			K = ys.shape[0]
			batch_size = batch_bboxes.shape[0]

			batch_center_idxs = []
			batch_gt_idxs = []
			batch_reg_targets = []
			batch_center_num = []
			batch_center_pts = []

			for i in range(batch_size):
				bboxes = batch_bboxes[i].clone()
				# bbox area
				areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
				# distance to four edges
				l = xs[:, None] - bboxes[:, 0][None]
				t = ys[:, None] - bboxes[:, 1][None]
				r = bboxes[:, 2][None] - xs[:, None]
				b = bboxes[:, 3][None] - ys[:, None]

				reg_targets = torch.stack([l, t, r, b], dim=2)
				# points inside the bboxes
				inside_mask = (reg_targets.min(dim=2)[0] >= 0) & (reg_targets.max(dim=2)[0] > 0)
				
				# select points 
				pts_inside_mask = inside_mask.sum(dim=1) > 0
				pts_inside_idxs = torch.where(pts_inside_mask)[0]
				num_inside = pts_inside_mask.sum().item()

				inside_reg_targets = reg_targets[pts_inside_mask]
				inside_xs = xs[pts_inside_mask]
				inside_ys = ys[pts_inside_mask]

				# points inside the center region of bboxes
				inside_center_mask = self.sample_region(target_centers[i].clone(), bboxes, inside_xs, inside_ys)	
				pts_inside_center_mask = inside_center_mask.sum(dim=1) > 0
				pts_inside_center_idxs = torch.where(pts_inside_center_mask)[0]
				num_center = pts_inside_center_idxs.shape[0]
				center_reg_targets = inside_reg_targets[pts_inside_center_mask]

				inside_center_mask = inside_center_mask[pts_inside_center_mask]
				
				# computing corresponding gt boxes			
				# handles situations where points are inside multiple boxes
				if self.center_mode == 'max':
					single_mask = inside_center_mask.sum(dim=1) == 1
					corr_gt_idx = batch_bboxes.new_zeros(num_center).long()
					corr_gt_idx[single_mask] = torch.where(inside_center_mask[single_mask])[1]
					
					# overlaps
					overlap_mask_copy = inside_center_mask.sum(dim=1) > 1
					overlap_mask = ~single_mask
					if overlap_mask.sum() > 0:
						overlap_reg_target = center_reg_targets[overlap_mask]
						left_right = overlap_reg_target[...,  [0, 2]]
						top_bottom = overlap_reg_target[..., [1, 3]]
						centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
									  (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
									  
						centerness[~inside_center_mask[overlap_mask]] = 0
						corr_gt_idx[overlap_mask] = torch.max(centerness, dim=1)[1]

				elif self.center_mode == 'area':
					if num_center > 0:
						inside_area = areas[None].repeat(num_center, 1)
						inside_area[inside_center_mask == 0] = self.INF
						corr_gt_idx = inside_area.min(dim=1)[1]
					else:
						corr_gt_idx = batch_bboxes.new_zeros(num_center).long()

				# regression target: l, t, r, b
				gt_reg_targets = center_reg_targets[torch.arange(num_center), corr_gt_idx, :]
				# indices of centering pts
				valid_center_mask = inside_center_mask.sum(dim=1) > 0
				valid_center_idxs = torch.where(valid_center_mask)[0]
				corr_gt_idx = corr_gt_idx[valid_center_idxs]

				# all pts -> center pts
				batch_center_idxs.append(pts_inside_idxs[pts_inside_center_mask])
				# inside gt idxs, inside reg_targets
				batch_gt_idxs.append(corr_gt_idx)
				batch_reg_targets.append(gt_reg_targets)
				batch_center_num.append(num_center)
				batch_center_pts.append(torch.stack((inside_xs[pts_inside_center_mask], inside_ys[pts_inside_center_mask]), dim=1))

			max_center_num = max(batch_center_num)
			# centerness branch: for all inside pixels
			output_centerness = batch_bboxes.new_zeros(batch_size, max_center_num)
			# other regression branches: for all center pixels
			output_center_idxs = batch_bboxes.new_zeros(batch_size, max_center_num)
			output_gt_idxs = batch_bboxes.new_zeros(batch_size, max_center_num)
			output_reg_targets = batch_bboxes.new_zeros(batch_size, max_center_num, 4)
			output_reg_mask = batch_bboxes.new_zeros(batch_size, max_center_num)
			output_center_pts = batch_bboxes.new_zeros(batch_size, max_center_num, 2)
			output_cls_map = batch_bboxes.new_zeros(batch_size, self.K, self.num_cls)

			for i in range(batch_size):
				center_num = batch_center_num[i]
				# inside idxs and center idxs
				global_center_idxs = batch_center_idxs[i].long()
				# cls branch
				gt_idxs = batch_gt_idxs[i].long()
				# idxs
				output_center_idxs[i, :center_num] = global_center_idxs
				# corresponding gt idxs
				output_gt_idxs[i, :center_num] = gt_idxs
				# 2d, 3d regression masks
				output_reg_mask[i, :center_num] = 1
				output_center_pts[i, :center_num, :] = batch_center_pts[i]
				reg_targets = batch_reg_targets[i]
				output_reg_targets[i, :center_num, :] = reg_targets
				output_cls_map[i, global_center_idxs, clses[i, gt_idxs].long()] = 1 

				if center_num > 0:	
					if self.target_center_mode == '2D':
						# compute centerness
						left_right = reg_targets[:, [0, 2]]
						top_bottom = reg_targets[:, [1, 3]]
						centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
									  (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
						centerness = torch.sqrt(centerness)
					else:
						center_dis = batch_center_pts[i] - target_centers[i][gt_idxs]
						direction_indi = (center_dis > 0).long() * 2
						hori_den = reg_targets[torch.arange(center_num), direction_indi[:, 0]]
						vert_den = reg_targets[torch.arange(center_num), direction_indi[:, 1] + 1]
						hori_len = center_dis[:, 0].abs() + hori_den
						vert_len = center_dis[:, 1].abs() + vert_den
						# both points and center on the boundary
						bound_mask = (hori_len == 0) | (vert_len == 0)
						centerness = center_dis.new_zeros(center_dis.shape[0])
						centerness[bound_mask] = 1
						centerness[~bound_mask] = hori_den[~bound_mask] / hori_len[~bound_mask] * vert_den[~bound_mask] / vert_len[~bound_mask]

					output_centerness[i, :center_num] = centerness 

			output_cls_map = output_cls_map.permute(0, 2, 1).reshape(batch_size, self.num_cls, self.output_height, self.output_width)

			targets = {'cls_map_2D': output_cls_map, 'reg_2D': output_reg_targets, 'centerness_2D': output_centerness}
			utils = {'center_pts_idxs': output_center_idxs.long(), 'corr_gt_idxs': output_gt_idxs.long(), 'reg_mask': output_reg_mask, 'center_pts': output_center_pts}

			return targets, utils

		def decode_box2d_fcos(self, centers, pred_offset, pad_size=None, out_size=None):
			box2d_center = centers.view(-1, 2)
			box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
			# left, top, right, bottom
			box2d[:, :2] = box2d_center - pred_offset[:, :2]
			box2d[:, 2:] = box2d_center + pred_offset[:, 2:]

			# max_obj x 4 for batch_size = 1
			if pad_size is not None:
				N = box2d.shape[0]
				out_size = out_size[0]
				# upscale and subtract the padding
				box2d = box2d * self.down_ratio - pad_size.repeat(1, 2)
				# clamp to the image bound
				box2d[:, 0::2].clamp_(min=0, max=out_size[0].item() - 1)
				box2d[:, 1::2].clamp_(min=0, max=out_size[1].item() - 1)

			return box2d

		def decode_box2d(self, centers, offset, dims, trans_mat=None, out_size=None):
			box2d_center = centers.view(-1, 2) + offset
			box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
			box2d[:, :2] = box2d_center - dims / 2
			box2d[:, 2:] = box2d_center + dims / 2

			# box2d[:, 0::2].clamp_(min=0, max=feat_size[0].item() - 1)
			# box2d[:, 1::2].clamp_(min=0, max=feat_size[1].item() - 1)

			# max_obj x 4 for batch_size = 1
			if trans_mat is not None:
				N = box2d.shape[0]
				out_size = out_size[0]
				# N x 3 x 3
				trans_mats_inv = trans_mat.inverse().type_as(box2d)
				# N x 3 x 1
				left_top_extend = torch.cat((box2d[:, :2], torch.ones(N, 1).type_as(box2d)), dim=1).unsqueeze(-1)
				right_bottom_extend = torch.cat((box2d[:, 2:], torch.ones(N, 1).type_as(box2d)), dim=1).unsqueeze(-1)

				left_top = torch.matmul(trans_mats_inv, left_top_extend).squeeze(-1)
				right_bottom = torch.matmul(trans_mats_inv, right_bottom_extend).squeeze(-1)
				box2d = torch.cat((left_top[:, :2], right_bottom[:, :2]), dim=1)

				box2d[:, 0::2].clamp_(min=0, max=out_size[0].item() - 1)
				box2d[:, 1::2].clamp_(min=0, max=out_size[1].item() - 1)

			return box2d

		def encode_projected_box3d(self, centers, offsets, pad_size, calibs, ):

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

			return torch.clamp(depth, min=self.depth_range[0], max=self.depth_range[1])

		def decode_location_flatten(self, points, offsets, depths, calibs, pad_size, batch_idxs):
			# all predictions are flatten, therefore kind of different from the function below
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

		def decode_uncertainty(self, pred_uncertainty):
			# clamp to [-2, 2]
			pred_uncertainty = torch.clamp(pred_uncertainty, min=self.uncertainty_range[0], max=self.uncertainty_range[1])
			# variance -> weights
			uncertainty_weights = torch.exp(- pred_uncertainty)
			# variance penalty
			uncertainty_loss = pred_uncertainty * 0.5

			return uncertainty_weights, uncertainty_loss

		def decode_depth_from_compact_keypoints(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
			# pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
			pred_height_3D = pred_dimensions[:, 1].clone()
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = pred_dimensions.new_zeros(pred_dimensions.shape[0])

			center_height = pred_keypoints[:, -2] - pred_keypoints[:, -1]
			corner_02_height = pred_keypoints[:, [4, 6]] - pred_keypoints[:, [8, 10]]
			corner_13_height = pred_keypoints[:, [5, 7]] - pred_keypoints[:, [9, 11]]

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

			return torch.stack([depth for depth in pred_keypoint_depths.values()], dim=1)

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

		def decode_vertical_edge_depth_from_keypoints(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
			# pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
			pred_height_3D = pred_dimensions[:, 1].clone()
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = pred_dimensions.new_zeros(pred_dimensions.shape[0])

			edge_heights = pred_keypoints[:, [0, 1, 2, 3, -2], 1] - pred_keypoints[:, [4, 5, 6, 7, -1], 1]
			all_edge_depths = []

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):			
				calib = calibs[idx]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				# edge depths
				edge_depths = calib.f_u * pred_height_3D[corr_pts_idx, None] / (F.relu(edge_heights[corr_pts_idx]) * self.down_ratio + self.EPS)
				all_edge_depths.append(edge_depths)

			# N x 5
			pred_depths = torch.clamp(torch.cat(all_edge_depths, dim=1), min=self.depth_range[0], max=self.depth_range[1])

			return pred_depths

		def decode_depth_from_pillars(self, pred_pillars, pred_dimensions, calibs, batch_idxs=None):
			# pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
			pred_height_3D = pred_dimensions[:, 1].clone()
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = pred_dimensions.new_zeros(pred_dimensions.shape[0])

			center_height = pred_pillars[:, -1]
			pillar_02_height = pred_pillars[:, [0, 2]]
			pillar_13_height = pred_pillars[:, [1, 3]]

			pred_pillar_depths = {'center': [], 'corner_02': [], 'corner_13': []}

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):			
				calib = calibs[idx]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				
				center_depth = calib.f_u * pred_height_3D[corr_pts_idx] / (F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
				corner_02_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(pillar_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
				corner_13_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(pillar_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)

				corner_02_depth = corner_02_depth.mean(dim=1)
				corner_13_depth = corner_13_depth.mean(dim=1)

				pred_pillar_depths['center'].append(center_depth)
				pred_pillar_depths['corner_02'].append(corner_02_depth)
				pred_pillar_depths['corner_13'].append(corner_13_depth)

			pred_pillar_depths = torch.stack([torch.cat(depth) for depth in pred_pillar_depths.values()], dim=1)
			pred_pillar_depths = torch.clamp(pred_pillar_depths, min=self.depth_range[0], max=self.depth_range[1])

			return pred_pillar_depths

		def decode_location(self,
							points,
							points_offset,
							depths,
							calibs,
							pad_size):
			'''
			retrieve objects location in camera coordinate based on projected points
			Args:
					points: projected points on feature map in (x, y)
					points_offset: project points offset in (delata_x, delta_y)
					depths: object depth z
					Ks: camera intrinsic matrix, shape = [N, 3, 3]
					trans_mats: transformation matrix from image to feature map, shape = [N, 3, 3]

			Returns:
					locations: objects location, shape = [N, 3]
			'''
			# number of points
			N = points_offset.shape[0]
			# batch size
			N_batch = len(calibs)
			max_obj = N // N_batch

			batch_id = torch.arange(N_batch).unsqueeze(1)
			obj_id = batch_id.repeat(1, max_obj).flatten()

			pad_size = pad_size[obj_id]

			points = points.view(-1, 2)
			assert points.shape[0] == N
			proj_points = points + points_offset
			proj_points = proj_points * self.down_ratio - pad_size
			proj_points = torch.cat((proj_points, depths[:, None]), dim=1)
			# transform image coordinates back to object locations
			locations = proj_points.new(N, 3).zero_()
			for i in range(N_batch):
				s = slice(i * max_obj, (i + 1) * max_obj)
				locations[s] = calibs[i].project_image_to_rect(proj_points[s])

			return locations

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

		def decode_orientation(self, vector_ori, locations):
			'''
			retrieve object orientation
			Args:
					vector_ori: local orientation in [sin, cos] format
					locations: object location

			Returns: for training we only need roty
							 for testing we need both alpha and roty

			'''

			locations = locations.view(-1, 3)
			rays = torch.atan2(locations[:, 0], locations[:, 2])
			alphas = torch.atan2(vector_ori[:, 0], vector_ori[:, 1])
			# alphas = alphas + 0.5 * PI

			# derive rotation_y
			rotys = alphas + rays + 0.5 * PI

			# in training time, it does not matter if angle lies in [-PI, PI]
			# it matters at inference time? todo: does it really matter if it exceeds.

			larger_idx = (rotys > PI).nonzero()
			small_idx = (rotys < -PI).nonzero()

			if len(larger_idx) != 0:
					rotys[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					rotys[small_idx] += 2 * PI

			return rotys, alphas


if __name__ == '__main__':
	pass