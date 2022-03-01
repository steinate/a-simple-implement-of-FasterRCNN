from __future__ import division
from collections import defaultdict
import itertools
import numpy as np

from bbox_tools import IOU


def calc_detection_voc_prec_rec(
    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difcs=None,
    iou_thresh=0.5
):
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difcs is None:
        gt_difcs = itertools.repeat(None)
    else:
        gt_difcs = iter(gt_difcs)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difc in \
        zip(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difcs):

        if gt_difc is None:
            gt_difcs = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            # 查找预测类别
            pr_mask = (pred_label == l) # 一张图上所有的框预测为类别l的为1其余为0的向量
            pred_bbox_l = pred_bbox[pr_mask] #
            pred_score_l = pred_score[pr_mask]

            # 排序
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            # 查找真实类别
            gt_mask = (gt_label == l)
            gt_bbox_l = gt_bbox[gt_mask]
            gt_difc_l = gt_difc[gt_mask] # 难易程度，一个gt框对应一个值

            n_pos[l] += np.logical_not(gt_difc_l).sum() # 一张图上所有的l类的框为容易的个数
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                # defaultdict(list, {0: [0, 0, 0], 1: [0, 0, 0], 2: [1, 1]})
                continue

            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = IOU(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            gt_index[iou.max(axis=1) < iou_thresh] = -1 # 如果iou小于阈值索引就设为-1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0],dtype=bool) # 一张图上gt框的数量,最初都未被选中
            # 每个预测框都有一个match对应
            # -1：标记为困难，0：iou小于阈值不匹配或者已经被选中过，1:行
            for idx in gt_index:
                if idx >= 0:
                    if gt_difc_l[idx]: # 困难模式不计入
                        match[l].append(-1)
                    else:
                        if not selec[idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[idx] = True # 第idx个框被选中
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1 # 最大类别数+1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys(): # 对于当前图片的每个类别
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)    # 真正例
        fp = np.cumsum(match_l == 0)    # 假正例

        prec[l] = tp / (fp + tp) # 查全率 (fp + tp):预测为正例的总数目
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l] # n_pos[l]是gt框的真实标签，代表图上真的有对应个l类的数目

    return prec, rec

def calc_detection_voc_ap(prec, rec, voc07=False):
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if voc07:
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    # 使用0代替数组x中的nan元素，使用有限的数字代替inf元素
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))
            # d = np.array([2, 0, 3, -4, -2, 7, 9])
            # np.maximum.accumulate(d)# array([2, 2, 3, 3, 3, 7, 9])
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap



def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboes, gt_labels, gt_difcs=None,
        iou_thresh=0.5, voc07 = False):
    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboes, gt_labels, gt_difcs, iou_thresh=iou_thresh
    )
    ap = calc_detection_voc_ap(prec, rec, voc07=voc07)

    return {'ap': ap, 'map': np.nanmean(ap)}