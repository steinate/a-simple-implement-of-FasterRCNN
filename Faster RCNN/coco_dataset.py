from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
import os
from PIL import Image
import random

class coco(Dataset):
    def __init__(self, data_dir, image_set, year):

        self.data_dir = data_dir
        self._year = year
        self._image_set = image_set
        self._data_name = image_set + year
        self._json_path = self._get_ann_file()

        # load COCO API
        self._COCO = COCO(self._json_path)

        with open(self._json_path) as anno_file:
            self.anno = json.load(anno_file)

        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])

        self.classes = self._classes
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats],
                                                   self._COCO.getCatIds())))

        self.coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                               self._class_to_ind[cls])
                                              for cls in self._classes[1:]])

    def __len__(self):
        return len(self.anno['images'])

    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 else 'image_info'
        return os.path.join(self.data_dir, 'annotations', prefix + '_' + self._image_set + self._year + '.json')

    def _image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = (str(index).zfill(12) + '.jpg')
        image_path = os.path.join(self.data_dir, self._data_name, file_name)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def __getitem__(self, idx):
        a = self.anno['images'][idx]
        image_idx = a['id']
        img_path = os.path.join(self.data_dir, self._data_name, self._image_path_from_index(image_idx))
        image = Image.open(img_path)

        width = a['width']
        height = a['height']

        annIds = self._COCO.getAnnIds(imgIds=image_idx, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)

        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        iscrowd = []
        for ix, obj in enumerate(objs):
            cls = self.coco_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            iscrowd.append(int(obj["iscrowd"]))

        # # convert everything into a torch.Tensor
        # image_id = torch.tensor([image_idx])
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # gt_classes = torch.as_tensor(gt_classes, dtype=torch.int32)
        # iscrowd = torch.as_tensor(iscrowd, dtype=torch.int32)

        return image, boxes, gt_classes, iscrowd

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    @property
    def class_to_coco_cat_id(self):
        return self._class_to_coco_cat_id


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
        self.Imgset = coco(opt.data_dir, 'train', '2014') # 初始化
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, id):
        img, bbox, label, difficult = self.Imgset.__getitem__(id)
        img, bbox, label, scale = self.tsf((img, bbox, label)) # transform
        return  img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.Imgset)

class TestDataset:
    '''
    img: -1~1
    ori_img: 0~255
    '''
    def __init__(self, opt):
        self.opt = opt
        self.Imgset = coco(opt.data_dir, 'val', '2014')

    def __getitem__(self, id):
        ori_img, bbox, label, difficult = self.Imgset.__getitem__(id)
        img = preprocess(ori_img)
        return  img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.Imgset)

