from __future__ import  absolute_import
from tqdm import tqdm
from torch.utils import data as data_

from config import device, opt
from data import Dataset, TestDataset, inverse_normalize

from faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import Trainer
import array_tools as at
# from vis_tools import visdom_bbox
from eval_tools import eval_detection_voc


def eval(dataloader, faster_rcnn):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficulties = list(), list(), list()
    for i ,(imgs, sizes, gt_bboxes_, gt_labels_, gt_difficulties_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficulties += list(gt_difficulties_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficulties,voc07=True
    )
    return result


if __name__ == '__main__':
    # 数据、模型准备
    #opt._parse() # 输出设置参数
    print('load data')
    dataset = Dataset(opt)
    # img, bbox, label, scale
    dataloader = data_.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=opt.num_workers
    )
    testset = TestDataset(opt)
    testdataloader = data_.DataLoader(
        testset,
        batch_size=1,
        num_workers=opt.test_num_workers,
        shuffle=False,
        # pin_memory=True
    )
    # print(dataset.__len__()) 5011
    # print(testdataloader.__len__()) 4952

    print('model construct')
    FasterRCNN = FasterRCNNVGG16()

    # 训练
    print('start training')
    trainer = Trainer(FasterRCNN).to(device)
    best_mAP = 0
    lr_ = opt.lr
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    # trainer.load('myargs/fasterrcnn')
    for epoch in range(opt.epoch):
        trainer.reset_meter() # 损失清零
        for i, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            # img: (1, C, H, W) H < 600, W < 1000
            # bbox: (1, N, 4)
            # label: (1, N)
            # scale: (1, )
            img, bbox, label, scale = img.to(device).float(), bbox_.to(device), label_.to(device), at.scalar(scale)
            trainer.train_step(img, bbox, label, scale)
            # if (i + 1) % opt.plot_every == 0:
            #     # loss
            #     trainer.vis.plot_many(trainer.get_meter_data())
            #
            #     # ground truth bbox
            #     ori_img = inverse_normalize(at.tonumpy(img[0]))
            #     gt_img = visdom_bbox(ori_img, at.tonumpy(bbox_[0]), at.tonumpy(label_[0]))
            #     trainer.vis.img('gt_img', gt_img)
            #
            #     # predict bbox
            #     _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img], visualize=True)
            #     pred_img = visdom_bbox(ori_img, at.tonumpy(_bboxes[0]), at.tonumpy(_labels[0]).reshape(-1), at.tonumpy(_scores[0]))
            #     trainer.vis.img('pred_img', pred_img)


        # 一个epoch结束
        eval_result = eval(testdataloader, FasterRCNN)
        print('epoch',epoch,' mAP:',eval_result['map'])
        # trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        # log_info = 'lr:{}, map:{}, loss:{}'.format(str(lr_), str(eval_result['map']),str(trainer.get_meter_data()))
        # trainer.vis.log(log_info)
        if eval_result['map'] > best_mAP:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)

        if (epoch + 1) % 10 == 0 :
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay





