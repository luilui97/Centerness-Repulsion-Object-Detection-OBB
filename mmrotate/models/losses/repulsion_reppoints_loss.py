# -*- coding: utf-8 -*-
# @Time    : 2022/12/1 14:49
# @Author  : Lei Gao
# @FileName: repulsion_reppoints_loss.py
# @Software: PyCharm
import torch
from mmdet.models.losses.utils import weighted_loss
from torch import nn
import numpy as np
from mmrotate.models.dense_heads.utils import points_center_pts
from mmcv.ops import min_area_polygons
from mmcv.ops import diff_iou_rotated_2d

from mmrotate.models.builder import ROTATED_LOSSES
from mmrotate.core.bbox.transforms import poly2obb, obb2xyxy
from mmcv.ops.diff_iou_rotated import oriented_box_intersection_2d, box2corners
import mmcv




@mmcv.jit
def attraction_loss(pred, target, mask):
    """
    Attraction term that a predicted box to approach its designated target.

    Args:
        pred (torch.Tensor): Convexes with shape (N, 18).
        target (torch.Tensor): Polygons with shape (N, 8).
        mask (torch.Tensor): mask for one GT

    Returns:
        torch.Tensor: attraction loss.
    """
    pred_min_area = min_area_polygons(pred)
    pred_obb = poly2obb(pred_min_area)
    target_obb = poly2obb(target)
    iou_list = []
    for i in range(mask.shape[0]):

        pred_i = pred_obb[mask[i],:].unsqueeze(0)
        target_i = target_obb[mask[i],:].unsqueeze(0)
        iou = diff_iou_rotated_2d(pred_i,target_i).reshape(-1)
        value, inds = iou.max(dim=0)
        iou_list.append(value)
    loss_attraction = 1 - torch.as_tensor(iou_list).cuda()

    return loss_attraction.mean()

def constraint_angle(x):
    return torch.where(
        torch.ge(x,torch.pi/2),
        torch.pi/2,
        x)

def smooth_ln(x, smooth):
    return torch.where(
        torch.le(x, smooth),
        -torch.log(1 - x),
        ((x - smooth) / (1 - smooth)) - np.log(1 - smooth)
    )

@weighted_loss
@mmcv.jit
def repGT_loss_single(pred, target, mask):
    """
    Other positive sample should be away from assigned GT
    Args:
        pred (torch.Tensor): Convexes with shape (N, 18).
        target (torch.Tensor): Polygons with shape (N, 8).

    Returns:
        torch.Tensor: repGT loss.
    """
    if pred.shape[0] <= 1:
        return torch.tensor(0).cuda().float()
    pred_min_area = min_area_polygons(pred)  # (N,8)
    pred_obb = poly2obb(pred_min_area).unsqueeze(0)  # (1,N,5)
    target_obb = poly2obb(target).unsqueeze(0)  # (1,N,5)
    pred_list = []
    GT_list = []
    for ind_pos in range(pred.shape[0]):
        if mask[ind_pos].sum() == 0:
            continue
        GT_list.append(target_obb[0, mask[ind_pos]])
        pred_list.append(pred_obb[0, ind_pos].repeat(mask[ind_pos].sum(), 1))
    if len(GT_list) == 0:
        return torch.tensor(0).float().cuda()
    P = torch.cat(pred_list).unsqueeze(0)
    GT = torch.cat(GT_list).unsqueeze(0)
    area_GT = GT[:, :, 2] * GT[:, :, 3]
    P_corners = box2corners(P)
    GT_corners = box2corners(GT)
    intersection, _ = oriented_box_intersection_2d(P_corners, GT_corners)
    IoG = intersection / area_GT
    RepGT_loss = smooth_ln(IoG, 0.5)
    return RepGT_loss




@mmcv.jit
def repBox_loss_single(pred, mask):
    """
    Positives should have minimal IoU among each other.

    Args:
        pred (torch.Tensor): Convexes with shape (N, 18).
    Returns:
        torch.Tensor: repGT loss.
    """
    if pred.shape[0] <= 1:
        return torch.tensor(0.0).cuda().float()
    pred_min_area = min_area_polygons(pred)  # (N,8)
    pred_obb = poly2obb(pred_min_area).unsqueeze(0)  # (1,N,5)
    Box1_list = []
    Box2_list = []
    for ind_pos in range(pred.shape[0]):
        if mask[ind_pos].sum() == 0:
            continue
        Box1_list.append(pred_obb[0, mask[ind_pos]])
        Box2_list.append(pred_obb[0, ind_pos].repeat(mask[ind_pos].sum(), 1))
    if len(Box1_list) == 0:
        return torch.tensor(0).float().cuda()
    Box1 = torch.cat(Box1_list).unsqueeze(0)
    Box2 = torch.cat(Box2_list).unsqueeze(0)
    IoU = diff_iou_rotated_2d(Box1, Box2).reshape(-1)
    smooth_IoU = smooth_ln(IoU, 0.5)
    return (smooth_IoU / torch.clamp(torch.sum(torch.gt(IoU, 0)).float(), min=1.0)).mean()


@ROTATED_LOSSES.register_module()
class RepulsionReppointsLoss(nn.Module):
    """
    Compute repulsion loss for oriented reppoints
    Args:
        loss_weight (float, optional): The weight of loss
    """

    def __init__(self, loss_weight=1.0, part_weight=None):
        super(RepulsionReppointsLoss, self).__init__()
        self.loss_weight = loss_weight
        self.part_weight = part_weight

    def forward(self,
                pts_pred_refine,
                bbox_gt,
                pos_inds,
                labels,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pts_pred_refine (torch.Tensor): Predicted convexes. (N, 18)
            bbox_gt (torch.Tensor): Corresponding gt convexes.
            pos_inds (torch.tensor): the  inds of  positive point set samples
            num_proposals_each_level (list[int]): proposals number of each level
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            loss (torch.Tensor)
        """
        pos_pts_pred_refine = pts_pred_refine[pos_inds]
        pos_bbox_gt = bbox_gt[pos_inds]

        # compute attraction loss

        with torch.no_grad():
            pos_labels = labels[pos_inds]
            target_pos_labels = pos_labels.unsqueeze(1)
            all_pos_labels = pos_labels.unsqueeze(0)
            cls_filter = target_pos_labels==all_pos_labels

            # filter redundant boxes
            gt_filter = filter_redundant_GT(pos_bbox_gt)
            gt_mask = get_pred_from_same_GT(pos_bbox_gt).unique(dim=0)
            # merge
            gt_filter &= cls_filter
            # change pred from (N,18) -> (N, 8) 9 points -> 4 points
            pred_min_area = min_area_polygons(pos_pts_pred_refine)
            # change pred from (N,8) -> (N,5) [x_ctr,y_ctr,w,h,angle]
            pred_obb = poly2obb(pred_min_area)
            # change pred from (N,5) -> (N,4) [x_lt,y_lt,x_rb,y_rb]
            pred_xyxy = obb2xyxy(pred_obb)
            # change gt from (N,8) -> (N,5)
            bbox_gt_obb = poly2obb(pos_bbox_gt)
            # change gt from (N,5) -> (N,4)
            bbox_gt_xyxy = obb2xyxy(bbox_gt_obb)
            # compute repGT iou
            inds_filter_GT = iou_filter(pred_xyxy, bbox_gt_xyxy)
            inds_filter_Box = iou_filter(pred_xyxy, pred_xyxy,is_same=True)
            inds_filter_GT &= gt_filter
            inds_filter_Box &= gt_filter
        loss_attraction = attraction_loss(pos_pts_pred_refine,pos_bbox_gt,gt_mask)
        loss_regGT = repGT_loss_single(pos_pts_pred_refine,
                                       pos_bbox_gt,
                                       mask=inds_filter_GT,
                                       reduction='mean'
                                       )*self.part_weight[1]
        loss_repBox = repBox_loss_single(pos_pts_pred_refine,
                                         mask=inds_filter_Box
                                         )*self.part_weight[2]
        # loss = loss_attraction + loss_regGT + loss_repBox
        # if torch.isnan(loss):
        #     print(loss_attraction, loss_regGT, loss_repBox)
        # return loss,
        return self.loss_weight*loss_attraction, self.loss_weight*loss_regGT, self.loss_weight*loss_repBox

def iou_filter(box1, box2, is_same=False):
    """
    Use Horizontal iou to filter pred and target
    Args:
        box1 (torch.Tensor): (N,4): x1,y1,x2,y2 (lt,rb)
        box2 (torch.Tensor): (N,4): x1,y1,x2,y2 (lt,rb)

    Returns:
        torch.Tensor: inds of boxes that iou>0

    """
    assert box1.shape[0] == box2.shape[0]
    IoU_hbb = calc_iou(box1, box2)
    IoU_fileter = IoU_hbb > 0
    mask = torch.full((box1.shape[0],), fill_value=True).to(IoU_hbb.device)
    mask_diag = ~torch.diag(mask)
    IoU_fileter &= mask_diag
    if is_same:
        return IoU_fileter.triu()
    return IoU_fileter
    # inds_row,_ = IoU_fileter.max(axis=0)
    # inds_column,_ = IoU_fileter.max(axis=1)
    # return inds_column&inds_column


def filter_redundant_GT(gt):
    """
    Remove redundant mask
    Args:
        gt (torch.Tensor): ground_truth_gt (N,8)

    Returns (torch.Tensor):
        mask matrix of each gt
    """
    target_gt = gt.unsqueeze(1)
    all_gt = gt.unsqueeze(0)
    mask = (target_gt != all_gt)
    mask = torch.any(mask,dim=2)
    return mask


def get_pred_from_same_GT(gt):
    """
    Get preds of same GT
    Args:
        gt (torch.Tensor): ground_truth_gt (N,8)

    Returns:
        mask matrix of each gt
    """
    target_gt = gt.unsqueeze(1)
    all_gt = gt.unsqueeze(0)
    mask = (target_gt == all_gt)
    mask = torch.all(mask,dim=2)
    return mask

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU


