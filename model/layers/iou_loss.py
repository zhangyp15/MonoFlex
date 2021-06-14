import torch
import math
import pdb
from torch import nn
from shapely.geometry import Polygon

class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        return losses, ious

def get_corners_torch(x, y, w, l, yaw):
    device = x.device
    bev_corners = torch.zeros((4, 2), dtype=torch.float, device=device)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners

def get_corners(bboxes):
    # bboxes: x, y, w, l, alpha; N x 5
    corners = torch.zeros((bboxes.shape[0], 4, 2), dtype=torch.float, device=bboxes.device)
    x, y, w, l = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # compute cos and sin
    cos_alpha = torch.cos(bboxes[:, -1])
    sin_alpha = torch.sin(bboxes[:, -1])
    # front left
    corners[:, 0, 0] = x - w / 2 * cos_alpha - l / 2 * sin_alpha
    corners[:, 0, 1] = y - w / 2 * sin_alpha + l / 2 * cos_alpha

    # rear left
    corners[:, 1, 0] = x - w / 2 * cos_alpha + l / 2 * sin_alpha
    corners[:, 1, 1] = y - w / 2 * sin_alpha - l / 2 * cos_alpha

    # rear right
    corners[:, 2, 0] = x + w / 2 * cos_alpha + l / 2 * sin_alpha
    corners[:, 2, 1] = y + w / 2 * sin_alpha - l / 2 * cos_alpha

    # front right
    corners[:, 3, 0] = x + w / 2 * cos_alpha - l / 2 * sin_alpha
    corners[:, 3, 1] = y + w / 2 * sin_alpha + l / 2 * cos_alpha

    return corners

def get_iou_3d(pred_corners, target_corners):
    """ 
    :param corners3d: (N, 8, 3) in rect coords  
    :param query_corners3d: (N, 8, 3)   
    :return: IoU 
    """
    A, B = pred_corners, target_corners
    N = A.shape[0]

    # init output
    iou3d = pred_corners.new(N).zero_().float()

    # for height overlap, since y face down, use the negative y
    min_h_a = - A[:, 0:4, 1].sum(dim=1) / 4.0
    max_h_a = - A[:, 4:8, 1].sum(dim=1) / 4.0
    min_h_b = - B[:, 0:4, 1].sum(dim=1) / 4.0
    max_h_b = - B[:, 4:8, 1].sum(dim=1) / 4.0

    # overlap in height
    h_max_of_min = torch.max(min_h_a, min_h_b)
    h_min_of_max = torch.min(max_h_a, max_h_b)
    h_overlap = torch.max(h_min_of_max.new_zeros(h_min_of_max.shape), h_min_of_max - h_max_of_min)

    # x-z plane overlap
    for i in range(N):
        bottom_a, bottom_b = Polygon(A[i, 0:4, [0, 2]]), Polygon(B[i, 0:4, [0, 2]])

        if bottom_a.is_valid and bottom_b.is_valid:
            # check is valid,  A valid Polygon may not possess any overlapping exterior or interior rings.
            bottom_overlap = bottom_a.intersection(bottom_b).area
        else:
            bottom_overlap = 0
            
        overlap3d = bottom_overlap * h_overlap[i]
        union3d = bottom_a.area * (max_h_a[i] - min_h_a[i]) + bottom_b.area * (max_h_b[i] - min_h_b[i]) - overlap3d
        iou3d[i] = overlap3d / union3d

    return iou3d
    