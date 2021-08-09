import torch
import torch.nn.functional as F
import pdb
from torch import nn

class Depth_Refine_Loss(nn.Module):
	def __init__(self, bin_num, bin_size, device):
		super(Depth_Refine_Loss, self).__init__()

		self.bin_size = bin_size
		self.bin_num = bin_num
		self.bin_offset = torch.arange(- bin_num / 2, bin_num / 2).ceil().to(device).view(1, -1) * bin_size

	def forward(self, pred_depths, pred_refine, target_depths):
		pred_refine_cls = pred_refine[:, :self.bin_num]
		pred_refine_reg = pred_refine[:, self.bin_num]

		pred_depths_plus_bin = pred_depths.view(-1, 1) + self.bin_offset
		depth_error = pred_depths_plus_bin - target_depths[:, None]
		target_bin_idxs = depth_error.abs().argmin(dim=1)

		# computing loss ...
		refine_cls_loss = F.cross_entropy(input=pred_refine_cls, target=target_bin_idxs, reduction='none')
		# clamp depth error
		depth_offset = torch.gather(depth_error, 1, target_bin_idxs[:, None]).clamp(min=-self.bin_size / 2, max=self.bin_size / 2).squeeze()
		refine_reg_loss = F.smooth_l1_loss(pred_refine_reg, depth_offset, reduction='none')

		return refine_cls_loss + refine_reg_loss * 10


class Berhu_Loss(nn.Module):
	def __init__(self):
		super(Berhu_Loss, self).__init__()
		# according to ECCV18 Joint taskrecursive learning for semantic segmentation and depth estimation
		self.c = 0.2

	def forward(self, prediction, target):
		pdb.set_trace()
		differ = (prediction - target).abs()
		c = torch.clamp(differ.max() * self.c, min=1e-4)
		# larger than c: l2 loss
		# smaller than c: l1 loss
		large_idx = (differ > c).nonzero()
		small_idx = (differ <= c).nonzero()

		loss = differ[small_idx].sum() + ((differ[large_idx] ** 2) / c + c).sum() / 2

		return loss

class Inverse_Sigmoid_Loss(nn.Module):
	def __init__(self):
		super(Inverse_Sigmoid_Loss, self).__init__()

	def forward(self, prediction, target, weight=None):
		trans_prediction = 1 / torch.sigmoid(target) - 1
		loss = F.l1_loss(trans_prediction, target, reduction='none')
		
		if weight is not None:
			loss = loss * weight

		return loss

class Log_L1_Loss(nn.Module):
	def __init__(self):
		super(Log_L1_Loss, self).__init__()

	def forward(self, prediction, target, weight=None):
		loss = F.l1_loss(torch.log(prediction), torch.log(target), reduction='none')
		
		if weight is not None:
			loss = loss * weight

		return loss

class Ins_Rela_Loss(nn.Module):
	def __init__(self):
		super(Ins_Rela_Loss, self).__init__()

	def forward(self, prediction, target, weight=None):
		loss = F.l1_loss(prediction, torch.log(target), reduction='none')
		
		if weight is not None:
			loss = loss * weight

		return loss.sum() 


if __name__ == '__main__':
	target = torch.ones(100) * 25
	prediction = target + torch.rand(target.shape) * 25

	prediction = prediction[torch.argsort((prediction - target).abs())]

	c = 0.2
	differ = (prediction - target).abs()
	c = torch.clamp(differ.max() * c, min=1e-4)
	print(c)
	# larger than c: l2 loss
	# smaller than c: l1 loss
	large_idx = (differ > c).nonzero()
	small_idx = (differ <= c).nonzero()

	differ[large_idx] = ((differ[large_idx] ** 2) / c + c) / 2

	import matplotlib.pyplot as plt

	plt.figure()
	plt.plot((prediction - target).abs(), differ)
	plt.show()


