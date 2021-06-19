from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle

def get_model_params(model, cfg):
	# default learning rate 
	optim_cfg = cfg.SOLVER
	base_lr = optim_cfg.BASE_LR
	
	params = []
	for key, value in model.named_parameters():
		if not value.requires_grad: continue

		# default learning rate
		key_lr = [base_lr]
		# bias learning rate 
		if "bias" in key: key_lr.append(cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR)
		params += [{"params": [value], "lr": max(key_lr)}]

	return params

def build_optimizer(model, cfg):
	optim_cfg = cfg.SOLVER

	if optim_cfg.OPTIMIZER != 'adam_onecycle':
		model_params = get_model_params(model, cfg)

	if optim_cfg.OPTIMIZER == 'adam':
		optimizer = optim.Adam(model_params, lr=optim_cfg.BASE_LR, weight_decay=optim_cfg.WEIGHT_DECAY, betas=(0.9, 0.99))

	elif optim_cfg.OPTIMIZER == 'adamw':
		optimizer = optim.AdamW(model_params, lr=optim_cfg.BASE_LR, weight_decay=optim_cfg.WEIGHT_DECAY, betas=(0.9, 0.99))
	
	elif optim_cfg.OPTIMIZER == 'sgd':
		optimizer = optim.SGD(
			model_params, lr=optim_cfg.BASE_LR, weight_decay=optim_cfg.WEIGHT_DECAY,
			momentum=optim_cfg.MOMENTUM
		)

	elif optim_cfg.OPTIMIZER == 'adam_onecycle':
		def children(m: nn.Module):
			return list(m.children())

		def num_children(m: nn.Module) -> int:
			return len(children(m))

		flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
		get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

		optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
		optimizer = OptimWrapper.create(
			optimizer_func, optim_cfg.BASE_LR, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
		)
	else:
		raise NotImplementedError

	return optimizer

def build_scheduler(optimizer, total_iters_each_epoch, optim_cfg, last_epoch=-1):
	decay_steps = optim_cfg.STEPS
	
	def lr_lbmd(cur_epoch):
		cur_decay = 1
		for decay_step in decay_steps:
			if cur_epoch >= decay_step:
				cur_decay = cur_decay * optim_cfg.LR_DECAY
		
		return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.BASE_LR)

	lr_warmup_scheduler = None
	total_steps = optim_cfg.MAX_ITERATION

	# OneCycle only works for fastai optimizer
	if optim_cfg.OPTIMIZER.find('onecycle') >= 0:
		lr_scheduler = OneCycle(
			optimizer, total_steps, optim_cfg.BASE_LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
		)
	else:
		lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

		if optim_cfg.LR_WARMUP:
			lr_warmup_scheduler = CosineWarmupLR(
				optimizer, T_max=optim_cfg.WARMUP_STEPS,
				eta_min=optim_cfg.BASE_LR / optim_cfg.DIV_FACTOR
			)

	return lr_scheduler, lr_warmup_scheduler
