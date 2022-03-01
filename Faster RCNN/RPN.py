import numpy as np
from torch.nn import functional as F
from torch import nn
from creator_tools import ProposalLayer

def normal_init(m, mean, std):
    m.weight.data.normal_(mean, std)
    m.bias.data.zero_()

def anchor_base_generator(base_size = 16, ratios = [0.5, 1, 2], anchor_scale = [8, 16, 32]):
    '''
    base_size: 边框基本长度
    ratios: 宽高比
    anchor_scale:边框长度放大比例
    以16*16映射为特征图上一个像素点，生成128*128、256*256和512*512的锚框
    '''
    x = base_size / 2
    y = base_size / 2
    anchor_base = np.zeros((len(ratios)*len(anchor_scale), 4),dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scale)):
            h = base_size * anchor_scale[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scale[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scale) + j
            anchor_base[index, 0] = y - h / 2
            anchor_base[index, 1] = x - w / 2
            anchor_base[index, 2] = y + h / 2
            anchor_base[index, 3] = x + w / 2
    return anchor_base

# t = anchor_generator():
# [[ -37.254833  -82.50967    53.254833   98.50967 ]
#  [ -82.50967  -173.01933    98.50967   189.01933 ]
#  [-173.01933  -354.03867   189.01933   370.03867 ]
#  [ -56.        -56.         72.         72.      ]
#  [-120.       -120.        136.        136.      ]
#  [-248.       -248.        264.        264.      ]
#  [ -82.50967   -37.254833   98.50967    53.254833]
#  [-173.01933   -82.50967   189.01933    98.50967 ]
#  [-354.03867  -173.01933   370.03867   189.01933 ]]

def anchor_generator(anchor_base, feat_stride, height, width):
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0] # 每个中心点的anchor数（9）
    K = shift.shape[0] # height * width
    # 一共 (height * width * 9) 个anchors

    anchor = anchor_base.reshape((1, A, 4)) + \
            shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor

class RegionProposalNetwork(nn.Module):

    def __init__(self,
                 in_channel=512,
                 mid_channel=512,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32],
                 feat_stride=16,
                 ):
        super(RegionProposalNetwork, self).__init__()
        # anchor_base: (9, 4)
        self.anchor_base = anchor_base_generator(
            anchor_scale=anchor_scales, ratios=ratios
        )
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalLayer(self)
        self.n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 3, 1, 1)
        self.score = nn.Conv2d(mid_channel, self.n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channel, self.n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, feature, img_size, scale=1.):
        # x: feature map, size = [1, 512, H/16, W/16]
        _, c, h, w = feature.shape
        # anchor: (h * w * 9, 4)
        anchor = anchor_generator(self.anchor_base, self.feat_stride, h, w)

        x = F.relu(self.conv1(feature)) # feature经过卷积形状不变与x相同
        # 生成位置偏移参量
        rpn_locs = self.loc(x) # (1, 4*9, h, w)
        # contiguous():断开两个向量的联系  view(n, -1, 4): -1表示自动推理，作用相当于reshape
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4) # (1, h, w, 4*9) -> (1, h*w*9, 4)

        # 生成预测分数
        rpn_scores = self.score(x) # (1, 2*9, h, w)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous() # (1, h, w, 2*9)
        rpn_softmax_scores = F.softmax(rpn_scores.view(1, h, w, self.n_anchor, 2), dim=4)  # (1, h, w, 9, 2)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous() # (1, h, w, 9)
        rpn_fg_scores = rpn_fg_scores.view(1, -1) # 前景概率
        rpn_scores = rpn_scores.view(1, -1, 2) # 前景+背景概率

        # anchors(≈20000):位置回归-削减边界-去除小区-保留高分-NMS:rois(<200)
        rois = self.proposal_layer(
            rpn_locs[0].cpu().data.numpy(), # .data==.detach
            rpn_fg_scores[0].cpu().data.numpy(),
            anchor, img_size,
            scale=scale
        )
        return rpn_locs, rpn_scores, rois, anchor
