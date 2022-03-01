from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from torch import nn
from torchnet.meter import AverageValueMeter

from creator_tools import AnchorTargetCreator, ProposalTargetCreator
import torch as t
import array_tools as at
# from vis_tools import Visualizer
from config import opt, device

loss = namedtuple('loss',['rpn_loc_loss', 'rpn_cls_loss',
                          'roi_loc_loss', 'roi_cls_loss', 'total_loss'])

class Trainer(nn.Module):
    def __init__(self, faster_rcnn):
        super(Trainer, self).__init__()
        self.faster_rcnn = faster_rcnn

        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma
        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self.optimizer = self.faster_rcnn.get_optimizer()

        # self.vis = Visualizer(env=opt.env)

        #loss._fields: ('rpn_loc_loss', 'rpn_cls_loss', 'roi_loc_loss', 'roi_cls_loss', 'total_loss')
        self.meters = {k: AverageValueMeter() for k in loss._fields}

    def forward(self, img, bboxes, labels, scale):
        # ================extractor================
        # img: (1, C, H, W) H < 600, W < 1000
        # bbox: (1, N, 4)
        # label: (1, N)
        # scale: (1, )
        _, _, H, W = img.shape
        img_size = (H, W)
        # extractor
        # features: (1, 512, H/16, W/16)
        features = self.faster_rcnn.extractor(img)

        # ===================RPN===================
        # anchors = H/16 * W/16 * 9 <= 20646
        # rois < 2000
        # rpn_locs: (1, anchors, 4)
        # rpn_scores: (1, anchors, 2)
        # rois: (rois, 4)
        # anchor: (anchors, 4)
        rpn_locs, rpn_scores, rois, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # =================proposer=================
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        # n_sample = 128
        # sample_roi: (n_sample, 4)
        # gt_roi_loc: (n_sample, 4)
        # gt_roi_label: (n_sample,) 0: bg, ~0: fg
        # bbox: torch.Tensor
        # label: torch.Tensor
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std
        )
        # =================ROIHead=================
        # roi_cls_loc: (128, 84)
        # roi_score: (128, 21)
        #sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
        )






        # ==================loss==================
        # RPN loss
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size
        )

        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = fast_rcnn_loc_loss(
            rpn_loc,        # 由rpn网络训练而来
            gt_rpn_loc,     # 根据gt推断出来
            gt_rpn_label.data,
            self.rpn_sigma
        )
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

        # ROI loss
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4) # [128, 21, 4]
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().to(device), at.totensor(gt_roi_label).long()] # [128, 4]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma
        )

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.to(device))

        # total losses
        losses = [10*rpn_loc_loss, rpn_cls_loss, 10*roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)] # 在losses后添加一个元素（总损失）

        return loss(*losses)

    def train_step(self, img, bboxes, labels, scale):
        # img: (1, C, H, W) H < 600, W < 1000
        # bbox: (1, N, 4)
        # label: (1, N)
        # scale: int elem
        self.optimizer.zero_grad()
        losses = self.forward(img, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_opitimizer=False, save_path=None, **kwargs):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        # save_dict['vis_info'] = self.vis.state_dict()

        if save_opitimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'myargs/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path) #去掉文件名，返回目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        # self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meter(self):
        for key, meter in self.meters.items():
            meter.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = (in_weight * (x - t)).abs()
    flag = (diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (diff - 0.5 / sigma2))
    return y.sum()

def fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).to(device) # gt_label must be tensor
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).to(device)] = 1
    loc_loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float()) # 归一化，除以正例
    return loc_loss

