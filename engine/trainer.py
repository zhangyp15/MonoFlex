import datetime
import logging
import time
import pdb
import os
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from engine.inference import inference
from utils import comm
from utils.metric_logger import MetricLogger
from utils.comm import get_world_size
from torch.nn.utils import clip_grad_norm_

def reduce_loss_dict(loss_dict):
	"""
	Reduce the loss dictionary from all processes so that process with rank
	0 has the averaged results. Returns a dict with the same fields as
	loss_dict, after reduction.
	"""
	world_size = get_world_size()
	if world_size < 2:
		return loss_dict
	with torch.no_grad():
		loss_names = []
		all_losses = []
		for k in sorted(loss_dict.keys()):
			loss_names.append(k)
			all_losses.append(loss_dict[k])

		all_losses = torch.stack(all_losses, dim=0)
		dist.reduce(all_losses, dst=0)

		reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}

	return reduced_losses

def do_eval(cfg, model, data_loaders_val, iteration, depth_method):
	eval_types = ("detection",)
	dataset_name = cfg.DATASETS.TEST[0]

	if cfg.OUTPUT_DIR:
		output_folder = os.path.join(cfg.OUTPUT_DIR, dataset_name, "inference_{}".format(iteration))
		os.makedirs(output_folder, exist_ok=True)

	# modify depth method
	model.heads.post_processor.output_depth = depth_method

	evaluate_metric, dis_ious = inference(
		model,
		data_loaders_val,
		dataset_name=dataset_name,
		eval_types=eval_types,
		device=cfg.MODEL.DEVICE,
		output_folder=output_folder,
	)
	comm.synchronize()

	return evaluate_metric, dis_ious

def do_train(
		cfg,
		distributed,
		model,
		data_loader,
		data_loaders_val,
		optimizer,
		scheduler,
		warmup_scheduler,
		checkpointer,
		device,
		arguments,
):
	logger = logging.getLogger("monoflex.trainer")
	logger.info("Start training")

	meters = MetricLogger(delimiter=" ", )
	max_iter = cfg.SOLVER.MAX_ITERATION
	start_iter = arguments["iteration"]

	# enable warmup
	if cfg.SOLVER.LR_WARMUP:
		assert warmup_scheduler is not None
		warmup_iters = cfg.SOLVER.WARMUP_STEPS
	else:
		warmup_iters = -1

	model.train()
	start_training_time = time.time()
	end = time.time()

	eval_depth_methods = cfg.TEST.EVAL_DEPTH_METHODS
	default_depth_method = cfg.MODEL.HEAD.OUTPUT_DEPTH

	if len(eval_depth_methods) == 0:
		eval_depth_methods.append(default_depth_method)

	if comm.get_local_rank() == 0:
		writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'writer/{}/'.format(cfg.START_TIME)))
		best_mAP = np.zeros(len(eval_depth_methods))
		best_iteration = 0
		eval_iteration = 0
		record_metrics = ['Car_bev_', 'Car_3d_']
	
	for data, iteration in zip(data_loader, range(start_iter, max_iter)):
		data_time = time.time() - end

		images = data["images"].to(device)
		targets = [target.to(device) for target in data["targets"]]

		loss_dict, log_loss_dict = model(images, targets)
		losses = sum(loss for loss in loss_dict.values())

		# reduce losses over all GPUs for logging purposes
		log_losses_reduced = sum(loss for key, loss in log_loss_dict.items() if key.find('loss') >= 0)
		meters.update(loss=log_losses_reduced, **log_loss_dict)
		
		optimizer.zero_grad()
		losses.backward()
		# clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_NORM_CLIP)
		optimizer.step()

		if iteration < warmup_iters:
			warmup_scheduler.step(iteration)
		else:
			scheduler.step(iteration)

		batch_time = time.time() - end
		end = time.time()
		meters.update(time=batch_time, data=data_time)

		iteration += 1
		arguments["iteration"] = iteration

		eta_seconds = meters.time.global_avg * (max_iter - iteration)
		eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

		if comm.get_rank() == 0:
			depth_errors_dict = {key: meters.meters[key].value for key in meters.meters.keys() if key.find('MAE') >= 0}
			writer.add_scalars('train_metric/depth_errors', depth_errors_dict, iteration)
			writer.add_scalar('stat/lr', optimizer.param_groups[0]["lr"], iteration)  # save learning rate

			for name, meter in meters.meters.items():
				if name.find('MAE') >= 0: continue
				if name in ['time', 'data']: writer.add_scalar("stat/{}".format(name), meter.value, iteration)
				else: writer.add_scalar("train_metric/{}".format(name), meter.value, iteration)

		if iteration % 10 == 0 or iteration == max_iter:
			logger.info(
				meters.delimiter.join(
					[
						"eta: {eta}",
						"iter: {iter}",
						"{meters}",
						"lr: {lr:.8f} \n",
					]
				).format(
					eta=eta_string,
					iter=iteration,
					meters=str(meters),
					lr=optimizer.param_groups[0]["lr"],
				)
			)

		if iteration % cfg.SOLVER.SAVE_CHECKPOINT_INTERVAL == 0:
			logger.info('iteration = {}, saving checkpoint ...'.format(iteration))
			if comm.get_rank() == 0:
				checkpointer.save("model_checkpoint", **arguments)
			
		if iteration == max_iter and comm.get_rank() == 0:
			checkpointer.save("model_final", **arguments)

		if iteration % cfg.SOLVER.EVAL_INTERVAL == 0:
			for idx, depth_method in enumerate(eval_depth_methods):
				
				if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
					cur_epoch = iteration // arguments["iter_per_epoch"]
					logger.info('epoch = {}, evaluate model on validation set with depth {}'.format(cur_epoch, depth_method))
				else:
					logger.info('iteration = {}, evaluate model on validation set with depth {}'.format(iteration, depth_method))
				
				result_dict, dis_ious = do_eval(cfg, model, data_loaders_val, iteration, depth_method)
				
				if comm.get_rank() == 0:
					# only record more accurate R40 results
					result_dict = result_dict[0]
					if len(result_dict) > 0:
						for key, value in result_dict.items():
							for metric in record_metrics:
								if key.find(metric) >= 0:
									threshold = key[len(metric) : len(metric) + 4]
									writer.add_scalar("eval_{}_{}/{}".format(depth_method, threshold, key), float(value), eval_iteration + 1)

					for key, value in dis_ious.items():
						writer.add_scalar("IoUs_{}/{}".format(key, depth_method), value, eval_iteration + 1)				

					# record the best model according to the AP_3D, Car, Moderate, IoU=0.7
					important_key = '{}_3d_{:.2f}/moderate'.format('Car', 0.7)
					eval_mAP = float(result_dict[important_key])
					if eval_mAP >= best_mAP[idx]:
						# save best mAP and corresponding iterations
						best_mAP[idx] = eval_mAP
						best_iteration = iteration
						checkpointer.save("model_moderate_best_{}".format(depth_method), **arguments)

						if cfg.SOLVER.EVAL_AND_SAVE_EPOCH:
							logger.info('epoch = {}, best_mAP = {:.2f}, updating best checkpoint for depth {}'.
											format(cur_epoch, eval_mAP, depth_method))
						else:
							logger.info('iteration = {}, best_mAP = {:.2f}, updating best checkpoint for depth {}'.
											format(iteration, eval_mAP, depth_method))

			# reset default depth method
			model.heads.post_processor.output_depth = default_depth_method
			
			eval_iteration += 1
			model.train()
			comm.synchronize()

	total_training_time = time.time() - start_training_time
	total_time_str = str(datetime.timedelta(seconds=total_training_time))
	if comm.get_rank() == 0:
		logger.info(
			"Total training time: {} ({:.4f} s / it), best model is achieved at iteration = {}".format(
				total_time_str, total_training_time / (max_iter), best_iteration,
			)
		)
