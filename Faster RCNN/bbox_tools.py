import numpy as np
import matplotlib.pyplot as plt

def loc2bbox(ori_bbox,loc):
    '''
    Decode bounding boxes from bounding box offsets and scales.
    已知源框和位置偏差，求出目标框
    '''
    if ori_bbox.shape[0] == 0:
        return np.zeros((0,4),dtype=loc.dtype)

    ori_bbox = ori_bbox.astype(ori_bbox.dtype, copy=False)
    ori_height = ori_bbox[:,2] - ori_bbox[:,0]
    ori_width = ori_bbox[:,3] - ori_bbox[:,1]
    ori_cy = ori_bbox[:,0] + 0.5 * ori_height
    ori_cx = ori_bbox[:,1] + 0.5 * ori_width

    t_y = loc[:,0::4]
    t_x = loc[:,1::4]
    t_h = loc[:,2::4]
    t_w = loc[:,3::4]
    # 论文中的tx,ty,th,tw,经过归一化

    cy = t_y * ori_height[:,np.newaxis] + ori_cy[:,np.newaxis]
    cx = t_x * ori_width[:, np.newaxis] + ori_cx[:, np.newaxis]
    h = np.exp(t_h) * ori_height[:,np.newaxis]
    w = np.exp(t_w) * ori_width[:,np.newaxis]

    dst_bbox = np.zeros(loc.shape,loc.dtype)
    dst_bbox[:, 0::4] = cy - 0.5 * h
    dst_bbox[:, 1::4] = cx - 0.5 * w
    dst_bbox[:, 2::4] = cy + 0.5 * h
    dst_bbox[:, 3::4] = cx + 0.5 * w

    return dst_bbox

def bbox2loc(ori_bbox, dst_bbox):
    '''
    已知源框框和参考框框求出其位置偏差
    Encodes the source and the destination bounding boxes to "loc".
    ori_bbox: anchor box
    dst_bbox: predicted box/ground truth box
    '''
    ori_height = ori_bbox[:,2] - ori_bbox[:,0]
    ori_width = ori_bbox[:, 3] - ori_bbox[:, 1]
    ori_cy = ori_bbox[:, 0] + 0.5 * ori_height
    ori_cx = ori_bbox[:, 1] + 0.5 * ori_width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_cy = dst_bbox[:, 0] + 0.5 * base_height
    base_cx = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(ori_height.dtype).eps
    ori_height = np.maximum(eps,ori_height)
    ori_width = np.maximum(eps,ori_width)
    # 防止÷0

    t_y = (base_cy - ori_cy) / ori_height
    t_x = (base_cx - ori_cx) / ori_width
    t_w = np.log(base_width / ori_width)
    t_h = np.log(base_height / ori_height)

    #transpose():调换数组的行列值的索引值，类似于求矩阵的转置
    loc = np.vstack((t_y,t_x,t_h,t_w)).transpose()

    return loc

def IOU(boxes1,boxes2):
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas