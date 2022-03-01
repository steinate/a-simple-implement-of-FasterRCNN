from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
from torchvision.ops import nms
from torch import nn
from torch.nn import functional as F

from data import preprocess
from config import opt, device
from bbox_tools import loc2bbox
import array_tools as at

def nogard(f):
    def new_f(*arg, **kargs):
        with t.no_grad():
            return f(*arg, **kargs)
    return new_f

class FasterRCNN(nn.Module):
    '''
    1.Feature extraction
    2.Region Proposal Networks
    3.Localization and Classification Heads
    '''
    def __init__(self, extractor, rpn, head,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    ):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

        self.use_preset('evaluate')

    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.): # x:四维数组

        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, anchor = \
            self.rpn(h, img_size, scale)
        roi_locs, roi_scores = self.head(
            h, rois)
        # roi_cls_locs：每个RoIs的偏移量  [300, 84]
        # roi_scores：每个RoIs的类别预测分数 [300, 21]
        # rois：region of interest的坐标 (300, 4)
        return roi_locs, roi_scores, rois

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('error')

    def _suppress(self, ori_bbox, ori_prob):
        bbox = list()
        label = list()
        score = list()
        for l in range(1, self.n_class): # l=0为背景
            bbox_l = ori_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = ori_prob[:, l]
            mask = prob_l > self.score_thresh
            bbox_l = bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(bbox_l, prob_l, self.nms_thresh) #如果两个候选框IOU大于该值就忽略
            bbox.append(bbox_l[keep].cpu().numpy())
            #先将其转换成cpu float-tensor随后再转到numpy格式。 numpy不能读取CUDA tensor需要将它转化为 CPU tensor
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @nogard   #函数修饰器
    def predict(self, imgs, sizes=None, visualize=False):
        # eval方法主要是针对某些在train和predict两个阶段会有不同参数的层。
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:] # (H, W) 变换前
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
             prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            # img[None]:增加一个维度 img.shape=(C,H,W) img[None].shape=(1,C,H,W) 变换后
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1] # 宽比：变换前比变换后
            roi_loc, roi_scores, rois = self(img, scale=scale)
            # batch size = 1, 直接相等
            roi_scores = roi_scores.data
            roi_loc = roi_loc.data
            roi = at.totensor(rois) / scale

            mean = t.Tensor(self.loc_normalize_mean).to(device).repeat(self.n_class)[None] # (1, 21*4)
            std = t.Tensor(self.loc_normalize_std).to(device).repeat(self.n_class)[None]
            roi_loc = (roi_loc * std + mean)
            roi_loc = roi_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_loc) #扩展到与roi_loc一样的规模
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1,4)),
                                at.tonumpy(roi_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)

            cls_bbox[:, 0::2] = cls_bbox[:, 0::2].clamp(min=0, max=size[0]) #高
            cls_bbox[:, 1::2] = cls_bbox[:, 1::2].clamp(min=0, max=size[1]) #宽

            prob = (F.softmax(at.totensor(roi_scores), dim=1))
            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer






