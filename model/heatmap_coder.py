import numpy as np
import pdb

from skimage import transform as trans

def get_transfrom_matrix(center_scale, output_size):
	center, scale = center_scale[0], center_scale[1]
	# todo: further add rot and shift here.
	src_w, src_h = scale
	dst_w, dst_h = output_size

	src = np.zeros((3, 2), dtype=np.float32)
	dst = np.zeros((3, 2), dtype=np.float32)

	src[0, :] = center
	src[1, :] = center - np.array([src_w * 0.5, 0])
	src[2, :] = center - np.array([0, src_h * 0.5])

	dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
	dst[1, :] = np.array([0, dst_h * 0.5])
	dst[2, :] = np.array([dst_w * 0.5, 0])

	get_matrix = trans.estimate_transform("affine", src, dst)
	matrix = get_matrix.params

	return matrix.astype(np.float32)

def affine_transform(point, matrix):

	point = point.reshape(-1, 2)
	point_exd = np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)

	new_point = np.matmul(point_exd, matrix.T)

	return new_point[:, :2].squeeze()

def gaussian_radius(height, width, min_overlap=0.7):
	a1 = 1
	b1 = (height + width)
	c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
	sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
	r1 = (b1 + sq1) / 2

	a2 = 4
	b2 = 2 * (height + width)
	c2 = (1 - min_overlap) * width * height
	sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
	r2 = (b2 + sq2) / 2

	a3 = 4 * min_overlap
	b3 = -2 * min_overlap * (height + width)
	c3 = (min_overlap - 1) * width * height
	sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
	r3 = (b3 + sq3) / 2

	return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
	m, n = [(ss - 1.) / 2. for ss in shape]
	y, x = np.ogrid[-m:m + 1, -n:n + 1]

	# generate meshgrid 
	h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
	h[h < np.finfo(h.dtype).eps * h.max()] = 0

	return h

def draw_gaussian_1D(edgemap, center, radius):
	diameter = 2 * radius + 1
	sigma = diameter / 6
	gaussian_1D = np.arange(-radius, radius + 1) 
	gaussian_1D = np.exp(- (gaussian_1D * gaussian_1D) / (2 * sigma * sigma))
	# 1D mask
	left, right = min(center, radius), min(len(edgemap) - center, radius + 1)
	masked_edgemap = edgemap[center - left : center + right]
	masked_gaussian = gaussian_1D[radius - left : radius + right]

	if min(masked_gaussian.shape) > 0 and min(masked_edgemap.shape) > 0:
		np.maximum(masked_edgemap, masked_gaussian, out=masked_edgemap)

	return edgemap

def draw_umich_gaussian(heatmap, center, radius, k=1, ignore=False):
	diameter = 2 * radius + 1
	gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

	x, y = int(center[0]), int(center[1])
	height, width = heatmap.shape[0:2]

	left, right = min(x, radius), min(width - x, radius + 1)
	top, bottom = min(y, radius), min(height - y, radius + 1)

	masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
	masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
	'''
	Assign all pixels within the area as -1. They will be further suppressed by heatmap from other 
	objects with the maximum. Therefore, these don't care objects won't influence other positive samples.
	'''
	
	if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
		if ignore:
			masked_heatmap[masked_heatmap == 0] = -1
		else:
			np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

	return heatmap

def draw_umich_gaussian_2D(heatmap, center, radius_x, radius_y, k=1):	
	diameter_x, diameter_y = 2 * radius_x + 1, 2 * radius_y + 1
	gaussian = ellip_gaussian2D((diameter_y, diameter_x), sigma_x=diameter_x / 6, sigma_y=diameter_y / 6)

	x, y = int(center[0]), int(center[1])
	height, width = heatmap.shape[0:2]

	left, right = min(x, radius_x), min(width - x, radius_x + 1)
	top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

	masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
	masked_gaussian = gaussian[radius_y - top:radius_y + bottom, radius_x - left:radius_x + right]
	
	if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
		np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

	return heatmap

def ellip_gaussian2D(shape, sigma_x, sigma_y):
	m, n = [(ss - 1.) / 2. for ss in shape]
	y, x = np.ogrid[-m:m + 1, -n:n + 1]

	# generate meshgrid 
	h = np.exp(-(x * x) / (2 * sigma_x * sigma_x) - (y * y) / (2 * sigma_y * sigma_y))
	h[h < np.finfo(h.dtype).eps * h.max()] = 0

	return h

def draw_ellip_gaussian(heatmap, center, box2d, ratio=0.5, k=1):
	# only for one-side
	bbox_width = min(center[0] - box2d[0], box2d[2] - center[0])
	bbox_height = min(center[1] - box2d[1], box2d[3] - center[1])
	
	radius_x, radius_y = int(bbox_width * ratio), int(bbox_height * ratio)
	diameter_x, diameter_y = 2 * radius_x + 1, 2 * radius_y + 1
	gaussian = ellip_gaussian2D((diameter_y, diameter_x), sigma_x=diameter_x / 6, sigma_y=diameter_y / 6)

	x, y = int(center[0]), int(center[1])
	height, width = heatmap.shape[0:2]

	left, right = min(x, radius_x), min(width - x, radius_x + 1)
	top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

	masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
	masked_gaussian = gaussian[radius_y - top:radius_y + bottom, radius_x - left:radius_x + right]
	
	if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
		np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

	return heatmap


