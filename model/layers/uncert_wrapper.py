import torch
import torch.nn as nn
import pdb

from utils.comm import get_world_size

def make_multitask_wrapper(cfg):
    multi_task_wrapper = MultiTaskLossesWrapper(
                            use_uncert=cfg.MODEL.HEAD.USE_UNCERTAINTY, 
                            reduce_loss=cfg.MODEL.REDUCE_LOSS_NORM,
                            keys=cfg.MODEL.HEAD.LOSS_NAMES,
                            uncertainty_keys=cfg.MODEL.HEAD.LOSS_UNCERTAINTY,
                        )

    return multi_task_wrapper

class MultiTaskLossesWrapper(nn.Module):
    def __init__(self, use_uncert=True, reduce_loss=True, keys=[], uncertainty_keys=[]):
        super(MultiTaskLossesWrapper, self).__init__()

        self.use_uncert = use_uncert
        self.keys = keys
        self.uncertainty_keys = [key for key, uncertainty in zip(keys, uncertainty_keys) if uncertainty]
        self.extend_keys = [key.replace('_loss', '') + '_w' for key in self.uncertainty_keys]
        
        self.uncertainty_task_num = len(self.uncertainty_keys)
        # init uncertainty
        if self.use_uncert:
            self.log_vars = nn.Parameter(torch.zeros((self.uncertainty_task_num)))
        else:
            # when uncertainty is not used, simply set them to tensors which require no grad
            self.log_vars = torch.zeros((self.uncertainty_task_num))

        # init loss weights
        self.world_size = get_world_size()
        self.reduce_loss = reduce_loss

    def forward(self, loss_dict):
        # for terms of losses, weight them with uncertainties
        for idx, key in enumerate(self.keys):
            if key in self.uncertainty_keys:
                # loss with uncertainty
                uncertainty_idx = self.uncertainty_keys.index(key)
                loss_dict[key] = loss_dict[key] * torch.exp(-self.log_vars[uncertainty_idx]) + self.log_vars[uncertainty_idx]

        weights = torch.exp(-self.log_vars).tolist()
        uncertainty = dict(zip(self.extend_keys, weights))

        return loss_dict, uncertainty

    def reset_weight(self):
        with torch.no_grad():
            weights = torch.exp(-self.log_vars)
            factor = self.uncertainty_task_num / weights.sum()
            normed_weights = weights * factor

            self.log_vars.data = - torch.log(normed_weights)