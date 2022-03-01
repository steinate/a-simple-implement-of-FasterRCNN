from __future__ import  absolute_import
from __future__ import  division
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


def resize_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox

def flip_bbox(bbox, size, x_flip=False):
    H, W = size
    bbox = bbox.copy()
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox

def random_flip(img):
    x_flip = random.choice([True, False])
    if x_flip:
        img = img[:, :, ::-1]
    return img, x_flip

def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

#了解一下VOC数据集
class VOCDataset:
    def __init__(self, data_dir, split='trainval',  #文件夹名
                 use_difficult=False, return_difficult=False,  #VOC数据集图片参数
                 ):
        list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split)
        ) #数据文件夹下打开trainval.txt
        # trainval: The union of training data and Validation data，表明为trainval的数据用于训练
        # strip(): 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        self.ids = [id_.strip() for id_ in open(list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        '''
        返回第i张样本图片+bounding boxes
        img: (C, H, W) elem:0~255
        bbox: (N, 4)  int
        label: (N,)
        difficult: (N,)
        '''
        id_ = self.ids[i]
        anno = ET.parse(os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            #Find all matching subelements by tag name or path.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox = obj.find('bndbox')
            bbox.append([
                int(bndbox.find(tag).text) - 1 # 从零开始
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')
            ])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool_).astype(np.uint8)

        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult

    __getitem__ = get_example

# data_dir = r'C:\Users\pjq\Desktop\deeplearning\simple-faster-rcnn-pytorch-master\data\VOCdevkit\VOC2007'
# db = VOCDataset(data_dir)
# img, bbox, label, difficult = db.get_example(6)
# for elem in label:
#     print(VOC_BBOX_LABEL_NAMES[int(elem)])
# print(label)
# img = img.transpose((1, 2, 0))/255
# plt.imshow(img)
# plt.axis('off')
# plt.show()

def normalize(img):
    normal = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    img = normal(t.from_numpy(img))
    return img.numpy()

def inverse_normalize(img):
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

def preprocess(img, min_size = 600, max_size = 1000):
    '''
    将图片进行最小最大化放缩然后进行归一化
    return appr -1~1 RGB
    '''
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    #return normalize(img)
    return normalize(img)

class Transform(object):
    '''
    图片进行缩放，使得长边小于等于1000，短边小于等于600(至少有一个等于)。
    对相应的bounding boxes也也进行同等尺度的缩放。
    对于Caffe的VGG16预训练模型，需要图片位于0-255，BGR格式，并减去一个均值，使得图片像素的均值为0。
    '''
    def __init__(self, min_size = 600, max_size = 1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img_data):
        img, bbox, label = img_data
        C, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        o_C, o_H, o_W = img.shape
        scale = o_H / H # scale对于H、W都一样
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))
        # 水平翻动
        img, x_flip = random_flip(img)
        bbox = flip_bbox(
            bbox, (o_H, o_W), x_flip
        )
        return img, bbox, label, scale


class Dataset:
    '''
    img: -1~1
    ori_img: 0~255
    '''
    def __init__(self, opt):
        self.opt = opt
        self.Imgset = VOCDataset(opt.data_dir) # 初始化
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, id):
        img, bbox, label, difficult = self.Imgset.get_example(id)
        img, bbox, label, scale = self.tsf((img, bbox, label)) # transform
        return  img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.Imgset)

class TestDataset:
    '''
    img: -1~1
    ori_img: 0~255
    '''
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.Imgset = VOCDataset(opt.data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, id):
        ori_img, bbox, label, difficult = self.Imgset.get_example(id)
        img = preprocess(ori_img)
        return  img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.Imgset) # 4952


# db = TestDataset(opt)
# img, ori_img, bbox, label, difficult = db.__getitem__(1)
# print(img)
# print(ori_img)
# for elem in label:
#     print(VOC_BBOX_LABEL_NAMES[int(elem)])
# img = img.transpose((1, 2, 0))
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# db = Dataset(opt)
# img, bbox, label, scale = db.__getitem__(1)
# for elem in label:
#     print(VOC_BBOX_LABEL_NAMES[int(elem)])
# img = img.transpose((1, 2, 0))
# plt.imshow(img)
# plt.axis('off')
# plt.show()
