import torch
import torch.nn as nn

class Grad_Clipper(nn.Module):
	def __init__(self, alpha, factor, norm_type=2.0):
		super(Grad_Clipper, self).__init__()
		# alpha for moving average
		self.alpha = alpha
		# clip norm factor
		self.factor = factor
		self.norm_type = norm_type
		self.grad_norm = -1

	def forward(self, model):
		model_grad_norm = self.get_grad_norm(model)

		# for initalization
		if self.grad_norm == -1:
			self.grad_norm = model_grad_norm

		# when the grad is larger than factor * mean grad
		if model_grad_norm > self.grad_norm * self.factor:
			torch.nn.utils.clip_grad_norm_(model.parameters(), self.factor * self.grad_norm)

		# update moving averaged grad norm
		self.grad_norm = self.grad_norm * self.alpha + (1 - self.alpha) * model_grad_norm

	def get_grad_norm(self, model):
		parameters = [p for p in model.parameters() if p.grad is not None]
		assert len(parameters) > 0
		device = parameters[0].grad.device
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), self.norm_type).to(device) for p in parameters]), self.norm_type)

		return total_norm

def grad_norm_(model, loss_dict, initial_loss, alpha):
	# loss dict -> loss_list
	task_losses = torch.cat([loss_dict[key] for key in sorted(loss_dict.keys())])
	num_task = task_losses.shape[0]
	weighted_loss = model.loss_weights * task_losses
	losses = weighted_loss.sum()
	# backward
	model.zero_grad()
	losses.backward(retain_graph=True)
	# task losses backward should not affect weights
	model.weights.grad.zero_()

	# compute gradient norms
	shared_layer = model.shared_layer
	grad_norm = []
	for i in range(num_task):
		grad = torch.autograd.grad(weighted_loss[i], shared_layer.parameters(), retain_graph=True)
		grad_norm.append(torch.norm(grad))
	grad_norm = torch.cat(grad_norm)
	
	# compute loss ratios and inverse training rate
	with torch.no_grad():
		loss_ratio = task_losses / initial_loss
		inverse_rate = loss_ratio / loss_ratio.mean()
		target_grad_norm = grad_norm.mean() * (inverse_rate ** alpha)

	# compute grad_norm loss
	gardnorm_loss = torch.sum(torch.abs(grad_norm - target_grad_norm))
	model.loss_weights.grad = torch.autograd.grad(gardnorm_loss, model.loss_weights)[0]