from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from RPN import RegionProposalNetwork
from faster_rcnn import FasterRCNN
import array_tools as at

def VGG16():
    model = vgg16(pretrained=True) # pretrained参数默认为False,ImgNet的参数
    features = list(model.features)[:-1] # 31层，去掉最后的Max pooling
    classifier = list(model.classifier)[:-1] # 去掉最后的线性层
    for layer in features[:10]: # 对于前十层不变
        for p in layer.parameters():
            p.requires_grad= False
    return nn.Sequential(*features), nn.Sequential(*classifier)

# feature =
# [Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#  Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#  Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#  Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),
#  Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#  ReLU(inplace=True),

# classifier=
# Sequential(
#   (0): Linear(in_features=25088, out_features=4096, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=4096, out_features=4096, bias=True)
#   (4): ReLU(inplace=True)
#   (5): Dropout(p=0.5, inplace=False)
#   (6): Linear(in_features=4096, out_features=1000, bias=True)
# )

class FasterRCNNVGG16(FasterRCNN):
    def __init__(self, n_class=20, feat_stride=16,# 不包括背景
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):
        extractor, classifier = VGG16()
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=feat_stride
        )
        head = RoiHead(
            n_class=n_class + 1, # 包括背景
            roi_size=7,
            spatial_scale=(1. / feat_stride),
            classifier=classifier
        )
        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head
        )

class RoiHead(nn.Module):
    # 包括背景（21类）
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(RoiHead, self).__init__()

        self.classifier = classifier
        self.loc = nn.Linear(4096, n_class * 4)
        self.scores = nn.Linear(4096, n_class)

        normal_init(self.loc, 0, 0.001)
        normal_init(self.scores, 0, 0.001)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale #采样比例
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)
        # ROI pooling具体操作如下：
        # （1）根据输入image，将ROI映射到feature map对应位置；
        # （2）将映射后的区域划分为相同大小的sections（sections数量与输出的维度相同）；
        # （3）对每个sections进行max pooling操作；


    def forward(self, x, rois):
        index = t.zeros(rois.shape[0]) # batch size=1,索引全为0
        index = at.totensor(index).float()
        rois = at.totensor(rois).float()
        rois_with_index = t.cat([index[:, None], rois], dim=1) # [:, None] 转置
        rois_with_index = rois_with_index[:, [0, 2, 1, 4, 3]]

        # torchvision.ops.roi_pool(input, boxes, output_size, spatial_scale=1.0)
        # input(Tensor[N, C, H, W]) – 输入张量
        # boxes(Tensor[K, 5] or List[Tensor[L, 4]]) – 输入的box坐标，格式：list(x1, y1, x2, y2)
        #                                             或者(batch_index, x1, y1, x2, y2)
        # output_size(int or Tuple[int, int]) – 输出尺寸, 格式： (height, width)
        # spatial_scale(float) – 将输入坐标映射到box坐标的尺度因子.默认: 1.0
        pool = self.roi(x, rois_with_index) # (128/300, 512, 7, 7)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_loc = self.loc(fc7)
        roi_scores = self.scores(fc7)
        return roi_loc, roi_scores

def normal_init(m, mean, std):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()