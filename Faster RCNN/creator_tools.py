import numpy as np
import torch
from torchvision.ops import nms

from config import device
from bbox_tools import loc2bbox,bbox2loc,IOU

def unmap(data, count, index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype = data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

class AnchorTargetCreator(object):
    '''
    从20000多个Anchor选出256个用于二分类和所有的位置回归！
    为预测值提供对应的真实值，选取的规则是：
    1.对于每一个Ground_truth bounding_box 从anchor中
    选取和它重叠度最高的一个anchor作为样本
    2.从剩下的anchor中选取和Ground_truth bounding_box重叠度
    超过0.7的anchor作为样本，注意正样本的数目不能超过128
    3.随机的从剩下的样本中选取和gt_bbox重叠度小于0.3的anchor
    作为负样本，正负样本之和为256
    '''
    def __init__(self,
                 n_sample=256,
                 pos_iou_threth=0.7,
                 neg_iou_threth=0.3,
                 pos_ratio=0.5):
        '''
        n_sample: 选出256个用于二分类和所有的位置回归
        pos_iou_threth: 选取和Ground_truth重叠度超过0.7的anchor作为正样本
        neg_iou_threth: 随机的从剩下的样本中选取和gt_bbox重叠度小于0.3的anchor作为负样本
        pos_ratio: 正负样本数量之比
        '''
        self.n_sample = n_sample
        self.pos_iou_threth = pos_iou_threth
        self.neg_iou_threth = neg_iou_threth
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        h, w = img_size
        n_anchor = len(anchor)
        # 位于图片内
        inside_index = np.where(
            (anchor[:, 0] >= 0) &
            (anchor[:, 1] >= 0) &
            (anchor[:, 2] <= h) &
            (anchor[:, 3] <= w)
        )[0]
        anchor = anchor[inside_index]
        argmax_ious, label = self.label_creator(anchor, bbox)
        loc = bbox2loc(anchor, bbox[argmax_ious])
        # 映射回去
        label = unmap(label, n_anchor, inside_index, fill=-1)
        loc = unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def label_creator(self, anchor, bbox):
        '''
        label: 1 is positive, 0 is negative, -1 is dont care
        '''
        label = np.empty((anchor.shape[0],), dtype=np.int32)
        label.fill(-1)

        # 匹配
        ious = IOU(anchor, bbox)  # anchor数×bbox数
        argmax_ious = ious.argmax(axis=1)  # 每个anchor对应的最大IOU索引
        max_ious = ious[np.arange(anchor.shape[0]), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)  # 第一个索引
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]  # 所有索引

        # 标记
        label[max_ious < self.neg_iou_threth] = 0
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_threth] = 1

        # 控制数量，削减
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size = (len(pos_index) - n_pos), replace = False)
            label[disable_index] = -1

        n_neg = self.n_sample - np.sum(label == 1) # n_pos
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

class ProposalLayer:
    '''
    对于每张图片，利用FeatureMap,计算H/16*W/16*9个anchor
    属于前景的概率和其对应的位置参数，即RPN网络正向作用的过程，
    然后从中选取概率较大的12000张，利用位置回归参数，
    修正这12000个anchor的位置4 利用非极大值抑制，选出2000个ROIS
    '''
    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # bounding box regression
        roi = loc2bbox(anchor, loc)
        # 调整边框在照片范围内
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        min_size = self.min_size * scale
        h = roi[:, 2] - roi[:, 0]
        w = roi[:, 3] - roi[:, 1]
        # 去除较小roi
        keep = np.where((h >= min_size) & (w >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]
        # 保留评分较高的roi
        order = score.ravel().argsort()[::-1] #建立从大到小的评分数组，ravel():将多维数组转换为一维数组
        order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]
        # 非极大值抑制
        keep = nms(
            torch.from_numpy(roi).to(device),
            torch.from_numpy(score).to(device),
            self.nms_thresh
        )
        keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi

class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample = 128,
                 pos_ratio = 0.25, #正样本的数量占比
                 pos_iou_threth = 0.5,
                 neg_iou_threth_hi = 0.5,
                 neg_iou_threth_lo = 0.0, #默认为0.1
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_threth = pos_iou_threth
        self.neg_iou_threth_hi = neg_iou_threth_hi
        self.neg_iou_threth_lo = neg_iou_threth_lo

    def __call__(self, roi, bbox, label,
                 loc_normal_mean=(0., 0., 0., 0.),
                 loc_normal_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape
        roi = np.concatenate((roi, bbox), axis=0)
        # np.concatenate: 拼接数组，axis=0，接下一行

        ideal_pos_num = np.round(self.n_sample * self.pos_ratio)
        iou = IOU(roi, bbox)  # N, K return (N, K)
        gt_assignment = iou.argmax(axis=1) # 每个roi对应最大iou索引
        max_iou = iou.max(axis=1) # 每个roi对应最大iou值

        gt_roi_label = label[gt_assignment] + 1 # 0 用于标识背景

        #选择正样例,大于阈值（满足一定比例）
        pos_index = np.where(max_iou >= self.pos_iou_threth)[0]
        num_of_pos = int(min(ideal_pos_num, pos_index.size))
        if pos_index.size > num_of_pos:
            pos_index = np.random.choice(pos_index, size=num_of_pos, replace=False) # replace=False: 不重复取样

        #选择负样例，iou处于某区间
        neg_index = np.where((max_iou < self.neg_iou_threth_hi) &
                             (max_iou >= self.neg_iou_threth_lo))[0]
        num_of_neg = self.n_sample - num_of_pos
        num_of_neg = int(min(num_of_neg, neg_index.size))
        if neg_index.size > num_of_neg:
            neg_index = np.random.choice(neg_index, size=num_of_neg, replace=False)

        keep = np.append(pos_index, neg_index) # <=128, 正例往往不够
        gt_roi_label = gt_roi_label[keep]
        gt_roi_label[num_of_pos:] = 0 #最后neg_index个即为负样例
        sample_roi = roi[keep]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normal_mean,np.float32))
                     / np.array(loc_normal_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label
