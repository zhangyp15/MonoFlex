import torch
from torch import nn
import pdb

from .detector_predictor import make_predictor
from .detector_loss import make_loss_evaluator
from .detector_infer import make_post_processor

class Detect_Head(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Detect_Head, self).__init__()

        self.predictor = make_predictor(cfg, in_channels)
        self.loss_evaluator = make_loss_evaluator(cfg)
        self.post_processor = make_post_processor(cfg)

    def forward(self, features, targets=None, test=False):
        x = self.predictor(features, targets)

        if self.training:
            loss_dict, log_loss_dict = self.loss_evaluator(x, targets)
            return loss_dict, log_loss_dict
        else:
            result, eval_utils, visualize_preds = self.post_processor(x, targets, test=test, features=features)
            return result, eval_utils, visualize_preds

def bulid_head(cfg, in_channels):
    
    return Detect_Head(cfg, in_channels)